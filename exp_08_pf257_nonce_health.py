"""
Experiment 08 - Nonce health suite.

Checks nonce uniqueness/independence properties and basic collision-risk
indicators under deterministic and random modes.
"""
from __future__ import annotations 

from typing import Any ,Dict ,List ,Optional ,Tuple 
import sys 
from pathlib import Path 
import hashlib 

import numpy as np 
import pandas as pd 

from pf257_experiment_base import BaseExperiment # type: ignore


# ---------------------------------------------------------------------------
# 9-bit popcount LUT: 0..511
# ---------------------------------------------------------------------------
_POPCOUNT_9 =np .array ([bin (i ).count ("1")for i in range (512 )],dtype =np .uint8 )


def _to_gray_u8 (img :np .ndarray )->np .ndarray :
    """See README.md for usage and details."""
    x =np .asarray (img )
    if x .ndim ==2 :
        return np .asarray (x ,dtype =np .uint8 ,order ="C")
    if x .ndim ==3 and x .shape [-1 ]in (3 ,4 ):
        g =np .mean (x [...,:3 ],axis =2 )
        return np .asarray (np .clip (np .rint (g ),0 ,255 ),dtype =np .uint8 ,order ="C")
    raise ValueError (f"Unsupported image shape: {x.shape}")


def _expected_cr_field9_bits ()->float :
    """See README.md for usage and details."""
    vals =np .arange (257 ,dtype =np .uint16 )# 0..256
    ps =[float (np .mean (((vals >>b )&1 )==1 ))for b in range (9 )]
    crs =[2.0 *p *(1.0 -p )for p in ps ]
    return float (np .mean (crs ))


class PF257NonceHealthSuite (BaseExperiment ):
    """Nonce health and collision-indicator checks (Experiment 08)."""

    DEFAULT_N_NONCES :int =10 
    DEFAULT_MC_NONCE_SAMPLES :int =100_000 
    STREAM_MAX_ELEMS :Optional [int ]=None # None -> H*W

    def __init__ (self ,config ,cipher ,**params :Any ):
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="PF257 nonce health: field stream independence + AEAD nonce binding",
        )

        self .N_NONCES =int (params .get ("n_nonces",self .DEFAULT_N_NONCES ))
        self .MC_NONCE_SAMPLES =int (params .get ("n_nonce_samples",self .DEFAULT_MC_NONCE_SAMPLES ))
        self .STREAM_MAX_ELEMS =params .get ("stream_max_elems",self .STREAM_MAX_ELEMS )

        self .EXPECTED_CR_FIELD9_BITS =float (_expected_cr_field9_bits ())

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _get_pf257_core (self )->Any :
        cw =self .cipher 
        adapter =getattr (cw ,"cipher",None )
        if adapter is None :
            raise AttributeError ("See README.md for usage and details.")
        core =getattr (adapter ,"core",None )
        return core if core is not None else adapter 

        # ------------------------------------------------------------------
        # Deterministik key32 (tekrarlanabilirlik)
        # ------------------------------------------------------------------
    @staticmethod 
    def _derive_key32 (tag :str ,seed :int )->bytes :
        return hashlib .sha256 (f"PF257_NONCE_HEALTH|{tag}|{seed}".encode ("utf-8")).digest ()

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _rng_for_image (seed :int ,img_name :str )->np .random .Generator :
        h =hashlib .sha256 (f"PF257_EXP08|{seed}|{img_name}".encode ("utf-8")).digest ()
        seed32 =int .from_bytes (h [:4 ],"big",signed =False )
        return np .random .default_rng (seed32 )

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _field_stream (core :Any ,key32 :bytes ,n_elems :int ,nonce :int )->np .ndarray :
        fn =getattr (core ,"_field_stream",None )
        if not callable (fn ):
            raise RuntimeError ("See README.md for usage and details.")
        out =fn (key32 ,int (n_elems ),int (nonce ))
        return np .asarray (out ,dtype =np .uint16 ,order ="C")

        # ------------------------------------------------------------------
        # Hamming mesafesi / oran (9-bit, 0x1FF)
        # ------------------------------------------------------------------
    @staticmethod 
    def _hamming_rate_bits_field9 (a :np .ndarray ,b :np .ndarray )->float :
        a16 =np .asarray (a ,dtype =np .uint16 ,order ="C")
        b16 =np .asarray (b ,dtype =np .uint16 ,order ="C")
        if a16 .shape !=b16 .shape :
            raise ValueError ("Shape mismatch in hamming computation.")

        xor9 =np .bitwise_and (np .bitwise_xor (a16 ,b16 ),np .uint16 (0x01FF ))
        diff_bits =int (_POPCOUNT_9 [xor9 ].sum ())
        total_bits =int (xor9 .size )*9 
        return float (diff_bits )/float (total_bits )if total_bits else 0.0 

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _aead_nonce_tamper_rejected (self ,img_u8 :np .ndarray ,nonce :int )->bool :
        """See README.md for usage and details."""
        enc_res =self .cipher .encrypt (img_u8 ,nonce =int (nonce ),return_priv =True )

        if not isinstance (enc_res ,tuple )or len (enc_res )<2 :
            raise RuntimeError ("See README.md for usage and details.")

        packet =enc_res [0 ]
        bob_priv =enc_res [1 ]

        if not isinstance (packet ,dict ):
            raise RuntimeError ("See README.md for usage and details.")

        tampered =dict (packet )
        tampered_nonce =int (tampered .get ("nonce",0 ))^0x1 
        tampered ["nonce"]=int (tampered_nonce )

        try :
            out =self .cipher .decrypt (tampered ,bob_priv ,verify =True )
            # NOTE: (comment removed; repository is English-only)
            if out is False :
                return True 
            return False 
        except Exception :
            return True 

            # ------------------------------------------------------------------
            # NOTE: (comment removed; repository is English-only)
            # ------------------------------------------------------------------
    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Starting PF257 Nonce Health Suite...")

        seed =int (getattr (self .config ,"seed",42 ))
        core =self ._get_pf257_core ()

        keyA =self ._derive_key32 ("A",seed )
        keyB =self ._derive_key32 ("B",seed +1 )

        # NOTE: (comment removed; repository is English-only)
        rng_mc =np .random .default_rng (seed )
        mc_nonces =rng_mc .integers (0 ,2 **63 -1 ,size =int (self .MC_NONCE_SAMPLES ),dtype =np .int64 )
        mc_unique =int (np .unique (mc_nonces ).size )
        mc_collisions =int (mc_nonces .size -mc_unique )
        mc_collision_rate =float (mc_collisions )/float (mc_nonces .size )if mc_nonces .size else 0.0 

        rows :List [Dict [str ,Any ]]=[]

        for img_name ,img in images .items ():
            img_u8 =_to_gray_u8 (img )
            H ,W =img_u8 .shape 
            n_elems =int (H *W )
            if self .STREAM_MAX_ELEMS is not None :
                n_elems =int (min (n_elems ,int (self .STREAM_MAX_ELEMS )))

                # NOTE: (comment removed; repository is English-only)
            rng_img =self ._rng_for_image (seed ,img_name )
            nonces =[int (rng_img .integers (0 ,2 **63 -1 ,dtype =np .int64 ))for _ in range (max (2 ,self .N_NONCES ))]

            nonces_unique =int (np .unique (np .asarray (nonces ,dtype =np .int64 )).size )
            nonces_collisions =int (len (nonces )-nonces_unique )
            nonces_collision_rate =float (nonces_collisions )/float (len (nonces ))if len (nonces )else 0.0 

            # NOTE: (comment removed; repository is English-only)
            streams =[self ._field_stream (core ,keyA ,n_elems ,n )for n in nonces ]
            rates_nonce :List [float ]=[
            self ._hamming_rate_bits_field9 (streams [i ],streams [i +1 ])for i in range (len (streams )-1 )
            ]

            cr_nonce_mean =float (np .mean (rates_nonce ))if rates_nonce else float ("nan")
            cr_nonce_min =float (np .min (rates_nonce ))if rates_nonce else float ("nan")
            cr_nonce_max =float (np .max (rates_nonce ))if rates_nonce else float ("nan")

            # NOTE: (comment removed; repository is English-only)
            s0 =self ._field_stream (core ,keyA ,n_elems ,nonces [0 ])
            sK =self ._field_stream (core ,keyB ,n_elems ,nonces [0 ])
            cr_key =float (self ._hamming_rate_bits_field9 (s0 ,sK ))

            # NOTE: (comment removed; repository is English-only)
            tamper_rejected =bool (self ._aead_nonce_tamper_rejected (img_u8 ,nonces [0 ]))

            rows .append ({
            "Image":img_name ,
            "H":int (H ),
            "W":int (W ),

            "Stream_elems":int (n_elems ),
            "Stream_bits_effective":int (n_elems )*9 ,

            "Nonces":int (len (nonces )),
            "Nonces_unique":int (nonces_unique ),
            "Nonces_collisions":int (nonces_collisions ),
            "Nonces_collision_rate":float (nonces_collision_rate ),

            "MC_nonce_samples":int (mc_nonces .size ),
            "MC_unique":int (mc_unique ),
            "MC_collisions":int (mc_collisions ),
            "MC_collision_rate":float (mc_collision_rate ),

            "Expected_CR_field9_bits":float (self .EXPECTED_CR_FIELD9_BITS ),

            "CR_nonce_mean":float (cr_nonce_mean ),
            "CR_nonce_min":float (cr_nonce_min ),
            "CR_nonce_max":float (cr_nonce_max ),
            "CR_key":float (cr_key ),

            "AEAD_nonce_tamper_rejected":bool (tamper_rejected ),
            })

            self .log (
            f"{img_name}: CR_nonce_mean={cr_nonce_mean:.6f}, CR_key={cr_key:.6f}, "
            f"img_nonce_coll_rate={nonces_collision_rate:.3e}, MC_coll_rate={mc_collision_rate:.3e}, "
            f"AEAD_nonce_tamper={'OK' if tamper_rejected else 'FAIL'}"
            )

        df =pd .DataFrame (rows )
        self .save_results (df ,"nonce_health_pf257.xlsx")
        self .log ("PF257 Nonce Health Suite completed.")
        return df 
