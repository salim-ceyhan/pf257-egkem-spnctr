"""
runner_pf257.py
Experiment runner for PF257-EGKEM-SPNCTR cipher analysis
(14-experiment suite; PF257 core + X25519 KEM compatibility adapter)

Overview
--------
This script executes the experimental suite used in the paper for the
PF257-EGKEM-SPNCTR construction. The cryptographic core implements an
ephemeral-static X25519 KEM-DEM flow and produces a serializable packet
containing the KEM header, wrapped parameters, ciphertext blocks, and an
optional HMAC tag (AEAD-like integrity).

Interfaces
----------
Core cipher (PF257_EGKEM_SPNCTR):
    encrypt(img, *, bob_pub_raw: bytes, nonce: int, debug_return_field_image: bool = False)
    decrypt(packet, *, bob_priv, verify: bool = True, bob_pub_raw: Optional[bytes] = None)

Compatibility layer (PF257CipherAdapter):
    The experimental framework expects a stable legacy-style API. Therefore,
    PF257CipherAdapter wraps the core cipher and exposes:
        encrypt(img, nonce: int, return_priv: bool = False, debug_return_intermediate: bool = False)
        decrypt(packet, bob_priv, verify: bool = True)

Implementation Notes
--------------------
- The adapter internally generates/holds the recipient (Bob) X25519 keypair
  when operating in self-contained experiment mode.
- Some experiments intentionally access internal adapter fields for diagnostic
  purposes and protocol binding checks:
      adapter.core
      adapter.bob_pub_h
      adapter.bob_priv_x

Reproducibility
---------------
- Output artifacts are written under RESULTS_PF257_PAPER/ by default.
- If deterministic mode is enabled, NumPy is seeded for experiment-level
  reproducibility; cryptographic randomness inside the core (secrets-based)
  remains intentionally non-deterministic unless explicitly overridden.
"""


from __future__ import annotations 

import sys 
from dataclasses import dataclass ,field 
from pathlib import Path 
from typing import Any ,Dict ,List ,Optional ,Type ,Union 

import numpy as np 
import pandas as pd 

from cryptography .hazmat .primitives .asymmetric import x25519 
from cryptography .hazmat .primitives import serialization 

from pf257_experiment_framework import (
CipherWrapper ,
CiphertextView ,
ExperimentConfig ,
ImageIO ,
validate_experiment_setup ,
)
from pf257_experiment_base import BaseExperiment 

# =======================================================================
# CORE (Prime Field + X25519-EGKEM)
# =======================================================================
from pf257_egkem_spnctr_v1_x25519 import PF257_EGKEM_SPNCTR 

# -----------------------------------------------------------------------
# Experiments (PF257 renamed modules/classes)
# -----------------------------------------------------------------------
from exp_00_pf257_lossless_verification import PF257LosslessVerification 
from exp_01_pf257_stage_metrics import PF257StageTripletMetrics 
from exp_02_pf257_histogram_analysis import PF257HistogramAnalysis 
from exp_03_pf257_differential_metrics import PF257DifferentialMetrics 
from exp_04_pf257_entropy_detailed import PF257EntropyDetailed 
from exp_05_pf257_randomness_nist import PF257RandomnessTests 
from exp_06_pf257_performance_breakdown import PF257PerformanceBenchmark 
from exp_07_pf257_ind_cpa_attack import PF257INDCpaHistogramExperiment 
from exp_08_pf257_nonce_health import PF257NonceHealthSuite 
from exp_09_pf257_layer_ablation import PF257LayerAblationStudy 
from exp_10_pf257_aead_tamper import PF257AEADTamperSuite 
from exp_11_pf257_comparison import PF257PerformanceAndMetricsComparison 
from exp_12_pf257_stream_integrity import PF257StreamIntegritySuite 
from exp_13_pf257_timing_leakage import PF257TimingLeakageSanity 


# -----------------------------------------------------------------------
# Experiment Meta
# -----------------------------------------------------------------------
@dataclass 
class ExperimentMeta :
    cls :Type [BaseExperiment ]
    description :str 
    params :Dict [str ,Any ]=field (default_factory =dict )


    # -----------------------------------------------------------------------
    # Registry (0-13)
    # -----------------------------------------------------------------------
EXPERIMENT_REGISTRY: Dict[int, ExperimentMeta] = {
    0: ExperimentMeta(
        cls=PF257LosslessVerification,
        description="Lossless encryption/decryption verification (pixel-perfect reconstruction).",
    ),
    1: ExperimentMeta(
        cls=PF257StageTripletMetrics,
        description="Stage-wise metrics for P/PF/C (entropy, chi-square p-values, neighbor correlations).",
    ),
    2: ExperimentMeta(
        cls=PF257HistogramAnalysis,
        description="Final-output histogram analysis (chi-square uniformity).",
    ),
    3: ExperimentMeta(
        cls=PF257DifferentialMetrics,
        description="Differential analysis (NPCR, UACI, AC, key sensitivity; pre/post sanity checks).",
    ),
    4: ExperimentMeta(
        cls=PF257EntropyDetailed,
        description="Information-leakage analysis (mutual information, local entropy).",
    ),
    5: ExperimentMeta(
        cls=PF257RandomnessTests,
        description="Keystream/mask randomness tests (NIST-inspired battery).",
    ),
    6: ExperimentMeta(
        cls=PF257PerformanceBenchmark,
        description="Performance breakdown (end-to-end and phase-wise timing).",
    ),
    7: ExperimentMeta(
        cls=PF257INDCpaHistogramExperiment,
        description="LR-IND-CPA^nr proxy experiment (histogram L1 distance and AUC).",
    ),
    8: ExperimentMeta(
        cls=PF257NonceHealthSuite,
        description="Nonce health suite (uniqueness, independence, collision probability).",
    ),
    9: ExperimentMeta(
        cls=PF257LayerAblationStudy,
        description="Layer ablation study (SPN-only, block-mixing, full pipeline; CTR-only side test).",
    ),
    10: ExperimentMeta(
        cls=PF257AEADTamperSuite,
        description="Tamper resistance under verify=True (bit flips, AD, structure, wrong key).",
    ),
    11: ExperimentMeta(
        cls=PF257PerformanceAndMetricsComparison,
        description="Baseline comparison: ours vs ChaCha20-12 and AES-CTR-256 (metrics and timing).",
    ),
    12: ExperimentMeta(
        cls=PF257StreamIntegritySuite,
        description="Stream/video integrity suite (sequential frames, bit-HD%, nonce uniqueness, replay prevention).",
    ),
    13: ExperimentMeta(
        cls=PF257TimingLeakageSanity,
        description="Timing leakage sanity check (content–time correlation).",
    ),
}

# =======================================================================
# COMPATIBILITY ADAPTER
# =======================================================================
BytesLike =Union [bytes ,bytearray ,memoryview ]

class PF257CipherAdapter :
    """See README.md for usage and details."""

    def __init__ (self ,bs :int ,*,aead :bool =True ,deterministic :bool =False ,seed :int =42 )->None :
        self .core =PF257_EGKEM_SPNCTR (bs =int (bs ),auth =bool (aead ))

        # NOTE: (comment removed; repository is English-only)
        if deterministic :
            self .bob_priv_x ,self .bob_pub_h =self ._deterministic_keypair (seed )
        else :
            self .bob_priv_x ,self .bob_pub_h =PF257_EGKEM_SPNCTR .kem_generate_keypair ()

    @staticmethod 
    def _priv_to_raw (priv :x25519 .X25519PrivateKey )->bytes :
        return priv .private_bytes (
        encoding =serialization .Encoding .Raw ,
        format =serialization .PrivateFormat .Raw ,
        encryption_algorithm =serialization .NoEncryption (),
        )

    @staticmethod 
    def _raw_to_priv (raw :BytesLike )->x25519 .X25519PrivateKey :
        b =bytes (raw )
        if len (b )!=32 :
            raise ValueError ("X25519 private key must be 32 raw bytes.")
        return x25519 .X25519PrivateKey .from_private_bytes (b )

    @staticmethod 
    def _deterministic_keypair (seed :int )->tuple [x25519 .X25519PrivateKey ,bytes ]:
        """See README.md for usage and details."""
        import hashlib 

        raw =hashlib .sha256 (f"PF257_BOB_PRIV|{int(seed)}".encode ("utf-8")).digest ()# 32 bytes
        priv =x25519 .X25519PrivateKey .from_private_bytes (raw )
        pub =priv .public_key ().public_bytes (
        encoding =serialization .Encoding .Raw ,
        format =serialization .PublicFormat .Raw ,
        )
        return priv ,pub 

    def encrypt (
    self ,
    img :np .ndarray ,
    *,
    nonce :int ,
    return_priv :bool =False ,
    debug_return_intermediate :bool =False ,
    **_ :Any ,
    ):
        packet ,out =self .core .encrypt (
        np .asarray (img ,dtype =np .uint8 ,order ="C"),
        bob_pub_raw =self .bob_pub_h ,# bytes (raw public key)
        nonce =int (nonce ),
        debug_return_field_image =bool (debug_return_intermediate ),
        )

        # NOTE: (comment removed; repository is English-only)
        if debug_return_intermediate :
            if isinstance (out ,np .ndarray )and out .dtype !=np .uint8 :
                out =CiphertextView .field2d_to_u8 (out .astype (np .uint16 ,copy =False ))

        if return_priv :
            return packet ,self ._priv_to_raw (self .bob_priv_x ),out 
        return packet ,out 

    def decrypt (
    self ,
    packet :dict ,
    *args :Any ,
    verify :bool =True ,
    **kwargs :Any ,
    )->np .ndarray :
        """See README.md for usage and details."""
        bob_priv =None 
        if len (args )>=1 :
            bob_priv =args [0 ]
        if "bob_priv"in kwargs and kwargs ["bob_priv"]is not None :
            bob_priv =kwargs ["bob_priv"]

        if bob_priv is None :
            priv =self .bob_priv_x 
        elif isinstance (bob_priv ,x25519 .X25519PrivateKey ):
            priv =bob_priv 
        elif isinstance (bob_priv ,(bytes ,bytearray ,memoryview )):
            priv =self ._raw_to_priv (bob_priv )
        else :
            raise TypeError ("bob_priv must be None, X25519PrivateKey, or 32-byte raw bytes.")

            # NOTE: (comment removed; repository is English-only)
        return self .core .decrypt (
        packet ,# type: ignore
        bob_priv =priv ,
        bob_pub_raw =self .bob_pub_h ,
        verify =bool (verify ),
        )


        # -----------------------------------------------------------------------
        # Runner
        # -----------------------------------------------------------------------
class ExperimentRunner :
    def __init__ (self ,config :ExperimentConfig ,image_dir :Path ):
        self .config =config 
        self .image_dir =Path (image_dir )

        if config .deterministic :
            np .random .seed (config .seed )

        self .cipher_raw =PF257CipherAdapter (
        bs =config .block_size ,
        aead =True ,
        deterministic =bool (config .deterministic ),
        seed =int (config .seed ),
        )
        self .cipher =CipherWrapper (self .cipher_raw )

        self .images =ImageIO .load_test_images (self .image_dir )
        validate_experiment_setup (config ,self .images )

        self .results :Dict [int ,pd .DataFrame ]={}

    def run_experiment (self ,exp_id :int )->Optional [pd .DataFrame ]:
        meta =EXPERIMENT_REGISTRY .get (exp_id )
        if not meta :
            print (f"⚠ Experiment {exp_id} not found in registry")
            return None 

        print ("\n"+"="*70 )
        print (f"EXPERIMENT {exp_id:02d}: {meta.description}")
        print ("="*70 )

        try :
            experiment =meta .cls (self .config ,self .cipher ,**meta .params )
            df =experiment .run (self .images )
            self .results [exp_id ]=df 
            print (f"✓ Experiment {exp_id:02d} completed successfully")
            print (f"  Output: {experiment.output_dir}")
            return df 
        except Exception as e :
            print (f"✗ Experiment {exp_id:02d} failed: {e}")
            import traceback 

            traceback .print_exc ()
            return None 

    def run_multiple (self ,exp_ids :List [int ])->None :
        print ("\n"+"="*70 )
        print (f"RUNNING {len(exp_ids)} EXPERIMENTS")
        print ("="*70 )

        for i ,exp_id in enumerate (exp_ids ,1 ):
            print (f"\n[{i}/{len(exp_ids)}] Starting Experiment {exp_id:02d}...")
            self .run_experiment (exp_id )

        self ._print_summary ()

    def _print_summary (self )->None :
        print ("\n"+"="*70 )
        print ("EXPERIMENT SUITE SUMMARY")
        print ("="*70 )

        if not self .results :
            print ("No experiments completed")
            return 

        print (f"Completed: {len(self.results)} experiments")
        print (f"Output directory: {self.config.output_dir}")
        print ("\nResults:")

        for exp_id in sorted (self .results .keys ()):
            df =self .results [exp_id ]
            meta =EXPERIMENT_REGISTRY .get (exp_id )
            desc =meta .description if meta else "???"
            print (f"  [{exp_id:02d}] {desc}")
            print (f"      → {len(df)} rows, {len(df.columns)} columns")

        print ("\n"+"="*70 )


def get_registered_experiments ()->List [int ]:
    return sorted (EXPERIMENT_REGISTRY .keys ())


def _print_banner ()->None :
    print (
    r"""
╔══════════════════════════════════════════════════════════════════╗
║   PF257-EGKEM-SPNCTR - Academic Experiment Suite                 ║
║   Prime-field block layer + X25519-EGKEM parameter wrapping      ║
╚══════════════════════════════════════════════════════════════════╝
"""
    )


if __name__ =="__main__":
    config =ExperimentConfig (
    block_size =16 ,
    deterministic =True ,
    seed =42 ,
    use_gpu =False ,
    output_dir =Path ("RESULTS_PF257_PAPER"),
    analysis_alphabet =256 ,
    )

    if len (sys .argv )>1 and sys .argv [1 ]=="list":
        print ("\nAvailable Experiments (PF257 Suite):")
        print ("-"*70 )
        for k ,meta in sorted (EXPERIMENT_REGISTRY .items ()):
            print (f"   [{k:2d}] {meta.description}")
        print ("-"*70 )
        sys .exit (0 )

    _print_banner ()

    repo_root =Path (__file__ ).resolve ().parent 
    image_dir =repo_root /"images"
    if not image_dir .exists ():
        print (f"✗ Image directory not found: {image_dir}")
        sys .exit (1 )

    try :
        runner =ExperimentRunner (config ,image_dir )
    except AssertionError as e :
        print (f"✗ Experiment setup validation failed: {e}")
        sys .exit (1 )
    except Exception as e :
        print(f"✗ Unexpected error during setup: {e}")
        import traceback 

        traceback .print_exc ()
        sys .exit (1 )

    experiments_to_run :List [int ]=[]

    try :
        cmd =sys .argv [1 ]if len (sys .argv )>1 else "all"

        if cmd =="single"and len (sys .argv )>2 :
            exp_id_to_run =int (sys .argv [2 ])
            if exp_id_to_run not in EXPERIMENT_REGISTRY :
                print (f"✗ Error: Experiment ID {exp_id_to_run} not found in registry.")
                print ("  Run with 'list' to see available experiments.")
                sys .exit (1 )
            experiments_to_run =[exp_id_to_run ]

        elif cmd =="essential":
            experiments_to_run =[0 ]

        elif cmd =="all":
            experiments_to_run =get_registered_experiments ()

        elif cmd not in ("list","single","essential","all"):
            print (f"Unknown command: {cmd}")
            print ("Usage: python runner_pf257.py [all|essential|list|single <id>]")
            sys .exit (1 )

    except Exception as e :
        print(f"✗ Argument parsing failed: {e}")
        sys.exit(1)

    if experiments_to_run :
        if len (experiments_to_run )==1 :
            exp_id =experiments_to_run [0 ]
            print (f"\n[1/1] Starting Experiment {exp_id:02d}...")
            runner .run_experiment (exp_id )
            runner ._print_summary ()
        else :
            runner .run_multiple (experiments_to_run )

        print ("\n✓ All experiments completed")
        print (f"   Results saved in: {config.output_dir.resolve()}")
    else :
        if cmd =="single":
            print("Usage: python runner_pf257.py single <id>")
            print("Tip: run `python runner_pf257.py list` to see available experiments.")
        else :
            print("Usage: python runner_pf257.py [all|essential|list|single <id>]")
