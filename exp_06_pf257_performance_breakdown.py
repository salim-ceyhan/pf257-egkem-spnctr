"""
Experiment 06 - Performance breakdown.

Measures end-to-end and phase-wise timing for encryption/decryption,
including pixel layer, block mixing, field masking, and integrity checks.
"""
from __future__ import annotations 

from dataclasses import dataclass 
from typing import Any ,Dict ,List ,Optional ,Tuple 
import sys 
from pathlib import Path 
import time 
import secrets 

import numpy as np 
import pandas as pd 

from pf257_experiment_base import BaseExperiment 


# ---------------------------------------------------------------------
# Helper: robust median / throughput
# ---------------------------------------------------------------------
def _median (xs :List [float ])->float :
    if not xs :
        return float ("nan")
    return float (np .median (np .asarray (xs ,dtype =np .float64 )))


def _mbps (nbytes :int ,seconds :float )->float :
    sec =max (float (seconds ),1e-12 )
    return float ((nbytes /(1024.0 *1024.0 ))/sec )


def _pad_shape (H :int ,W :int ,bs :int )->Tuple [int ,int ]:
    Hp =int (H +((-H )%bs ))
    Wp =int (W +((-W )%bs ))
    return Hp ,Wp 


def _sync_if_needed (*arrays :Any )->None :
    """See README.md for usage and details."""
    try :
        import cupy as cp # type: ignore
    except Exception :
        return 

    for a in arrays :
        try :
            if isinstance (a ,cp .ndarray ):
                cp .cuda .Stream .null .synchronize ()
                return 
        except Exception :
            continue 


def _time_call (fn ,*args ,**kwargs )->Tuple [float ,Any ]:
    """See README.md for usage and details."""
    t0 =time .perf_counter_ns ()
    out =fn (*args ,**kwargs )
    t1 =time .perf_counter_ns ()
    return (t1 -t0 )*1e-9 ,out 


    # ---------------------------------------------------------------------
    # NOTE: (comment removed; repository is English-only)
    # ---------------------------------------------------------------------
@dataclass (frozen =True )
class PhaseParams :
    S :np .ndarray 
    Kf :int 
    Kr :int 
    key32 :bytes 
    gk_vec :np .ndarray # uint16


class PF257PerformanceBenchmark (BaseExperiment ):
    """Performance timing breakdown (Experiment 06)."""

    DEFAULT_WARMUPS :int =3 
    DEFAULT_RUNS :int =10 
    DEFAULT_PHASE_RUNS :Optional [int ]=None # NOTE: (comment removed; repository is English-only)

    def __init__ (self ,config ,cipher ,**params :Any )->None :
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="PF257 performance benchmark (E2E encrypt/decrypt + phase breakdown)",
        )

        self .WARMUPS =int (params .get ("warmups",self .DEFAULT_WARMUPS ))
        self .RUNS =int (params .get ("runs",self .DEFAULT_RUNS ))
        pr =params .get ("phase_runs",self .DEFAULT_PHASE_RUNS )
        self .PHASE_RUNS =int (pr )if pr is not None else None 

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _get_pf257_core_ctx (self ):
        """See README.md for usage and details."""
        cw =self .cipher 
        adapter =getattr (cw ,"cipher",None )
        if adapter is None :
            raise AttributeError ("See README.md for usage and details.")
        core =getattr (adapter ,"core",None )
        if core is None :
            core =adapter 
        return core ,adapter 

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _unpack_encrypt_result (enc_res ):
        """See README.md for usage and details."""
        if not isinstance (enc_res ,tuple ):
            raise RuntimeError ("See README.md for usage and details.")

        if len (enc_res )==3 :
            pkt ,bob_priv ,out =enc_res 
            return pkt ,bob_priv ,out 

        if len (enc_res )==2 :
            pkt ,second =enc_res 
            # second bob_priv veya out olabilir. Bu experiment return_priv=True ile bob_priv bekler.
            return pkt ,second ,None 

        raise RuntimeError (f"See README.md for usage and details.")

    def _warmup_e2e (self ,img_u8 :np .ndarray ,n :int )->None :
        for _ in range (int (n )):
            nonce64 =secrets .randbits (64 )
            enc_res =self .cipher .encrypt (img_u8 ,nonce =int (nonce64 ),return_priv =True )
            pkt ,bob_priv ,_ =self ._unpack_encrypt_result (enc_res )
            # NOTE: (comment removed; repository is English-only)
            _ =self .cipher .decrypt (pkt ,bob_priv =bob_priv ,verify =True )
            _ =self .cipher .decrypt (pkt ,bob_priv =bob_priv ,verify =False )

    def _bench_e2e (self ,img_u8 :np .ndarray ,runs :int )->Tuple [float ,float ,float ,float ,float ]:
        """See README.md for usage and details."""
        enc_totals :List [float ]=[]
        dec_totals_v :List [float ]=[]
        dec_totals_nv :List [float ]=[]
        verify_only_times :List [float ]=[]

        for _ in range (int (runs )):
            nonce64 =secrets .randbits (64 )

            t_enc ,enc_res =_time_call (self .cipher .encrypt ,img_u8 ,nonce =int (nonce64 ),return_priv =True )
            pkt ,bob_priv ,_ =self ._unpack_encrypt_result (enc_res )
            enc_totals .append (t_enc )

            t_v ,_ =_time_call (self .cipher .decrypt ,pkt ,bob_priv =bob_priv ,verify =True )
            dec_totals_v .append (t_v )

            t_nv ,_ =_time_call (self .cipher .decrypt ,pkt ,bob_priv =bob_priv ,verify =False )
            dec_totals_nv .append (t_nv )

            # NOTE: (comment removed; repository is English-only)
            try :
                t_vo ,_ =_time_call (self .cipher .decrypt ,pkt ,bob_priv =bob_priv ,verify =True ,verify_only =True )
                verify_only_times .append (t_vo )
            except TypeError :
            # verify_only parametresi desteklenmiyor
                pass 

        med_enc =_median (enc_totals )
        med_v =_median (dec_totals_v )
        med_nv =_median (dec_totals_nv )
        aead_over =max (med_v -med_nv ,0.0 )
        med_verify_only =_median (verify_only_times )if verify_only_times else float ("nan")
        return med_enc ,med_v ,med_nv ,aead_over ,med_verify_only 

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _sample_phase_params (self ,core )->PhaseParams :
        if not (hasattr (core ,"_pack_params")and hasattr (core ,"_unpack_params")):
            raise AttributeError ("See README.md for usage and details.")

        seed16 =secrets .token_bytes (16 )
        Kf =int (secrets .randbelow (256 ))
        Kr =int (secrets .randbelow (256 ))
        key32 =secrets .token_bytes (32 )

        bs =int (getattr (core ,"bs",16 ))
        if bs <=0 :
            raise ValueError (f"See README.md for usage and details.")

            # NOTE: (comment removed; repository is English-only)
        gk =[int (secrets .randbelow (255 )+1 )for _ in range (bs )]

        plain =core ._pack_params (seed16 ,int (Kf ),int (Kr ),key32 ,gk )
        S ,_INV ,Kf2 ,Kr2 ,key32_2 ,gk_vec =core ._unpack_params (plain )

        if int (Kf2 )!=int (Kf )or int (Kr2 )!=int (Kr )or key32_2 !=key32 :
            raise RuntimeError ("Phase params unpack mismatch.")

        return PhaseParams (
        S =np .asarray (S ),
        Kf =int (Kf ),
        Kr =int (Kr ),
        key32 =key32 ,
        gk_vec =np .asarray (gk_vec ,dtype =np .uint16 ),
        )

    def _vu_from_gk (self ,core ,gk_vec :np .ndarray ):
        if hasattr (core ,"_vu_from_gk"):
            return core ._vu_from_gk (np .asarray (gk_vec ,dtype =np .uint16 ,order ="C"))
        if hasattr (core ,"_vu"):
            return core ._vu (np .asarray (gk_vec ,dtype =np .uint16 ,order ="C"))
        raise AttributeError ("See README.md for usage and details.")

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _time_pixel_stage (self ,core ,img_u8 :np .ndarray ,S :np .ndarray ,Kf :int ,Kr :int )->Tuple [float ,np .ndarray ]:
        if not hasattr (core ,"_encrypt_pixels"):
            raise AttributeError ("See README.md for usage and details.")
        t ,out =_time_call (core ._encrypt_pixels ,img_u8 ,S ,int (Kf ),int (Kr ))
        _sync_if_needed (out )
        return t ,np .asarray (out ,dtype =np .uint8 ,order ="C")

    def _time_block_mix (self ,core ,c2_u8 :np .ndarray ,gk_vec :np .ndarray ,Hp :int ,Wp :int )->Tuple [float ,np .ndarray ]:
        for fn_name in ("_pad_edge_u8","_blocks_of","_unblocks","_mix_affine"):
            if not hasattr (core ,fn_name ):
                raise AttributeError (f"See README.md for usage and details.")

        bs =int (getattr (core ,"bs",16 ))

        def _do ():
        # pad + field cast (0..255)
            c2_pad =core ._pad_edge_u8 (np .asarray (c2_u8 ,dtype =np .uint8 ,order ="C"),int (Hp ),int (Wp ))
            c2_field =np .asarray (c2_pad ,dtype =np .uint16 ,order ="C")

            blk4 =core ._blocks_of (c2_field )# (ni,nj,bs,bs)
            ni ,nj ,_ ,_ =blk4 .shape 
            n_blocks =int (ni *nj )

            blk_flat =blk4 .reshape ((n_blocks ,bs ,bs ))
            V ,U =self ._vu_from_gk (core ,np .asarray (gk_vec ,dtype =np .uint16 ,order ="C"))

            mixed_flat =core ._mix_affine (blk_flat ,V ,U )# (N,bs,bs) 0..256
            mixed_blk =mixed_flat .reshape ((ni ,nj ,bs ,bs ))
            mixed_field =core ._unblocks (mixed_blk ,int (Hp ),int (Wp ))# (Hp,Wp) uint16
            return mixed_field 

        t ,out =_time_call (_do )
        _sync_if_needed (out )
        return t ,np .asarray (out ,dtype =np .uint16 ,order ="C")

    def _time_stream_mask (self ,core ,mixed_field :np .ndarray ,key32 :bytes ,nonce :int )->Tuple [float ,np .ndarray ]:
        for fn_name in ("_field_stream","_blocks_of"):
            if not hasattr (core ,fn_name ):
                raise AttributeError (f"See README.md for usage and details.")

        Hp ,Wp =map (int ,mixed_field .shape )
        p =int (getattr (core ,"P_FIELD",257 ))

        def _do ():
            stream =core ._field_stream (key32 ,int (Hp *Wp ),int (nonce ))
            stream =np .asarray (stream ,dtype =np .int32 ,order ="C").reshape (Hp ,Wp )
            masked =(mixed_field .astype (np .int32 ,copy =False )+stream )%int (p )
            masked_field =masked .astype (np .uint16 ,copy =False )
            # pack: blocks_of
            cblk =core ._blocks_of (masked_field )
            return cblk 

        t ,out =_time_call (_do )
        _sync_if_needed (out )
        return t ,np .asarray (out ,dtype =np .uint16 ,order ="C")

        # ------------------------------------------------------------------
        # Main
        # ------------------------------------------------------------------
    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("PF257 performance benchmark started (E2E + phase breakdown).")

        core ,_adapter =self ._get_pf257_core_ctx ()

        rows :List [Dict [str ,Any ]]=[]
        phase_runs =int (self .PHASE_RUNS )if self .PHASE_RUNS is not None else int (self .RUNS )

        # NOTE: (comment removed; repository is English-only)
        for name in sorted (images .keys ()):
            img =images [name ]
            self .log (f"Processing {name}...")

            img_u8 =np .asarray (img ,dtype =np .uint8 ,order ="C")
            if img_u8 .ndim !=2 :
                raise ValueError (f"See README.md for usage and details.")

            H ,W =map (int ,img_u8 .shape )

            bs =int (getattr (core ,"bs",16 ))
            Hp ,Wp =_pad_shape (H ,W ,bs )

            # Byte accounting
            nbytes_orig =int (H *W )# uint8 bytes
            nbytes_pad_u8 =int (Hp *Wp )# uint8-equivalent
            nbytes_pad_u16 =int (Hp *Wp *2 )# field bytes (uint16)

            # ---------------- warmups (E2E) ----------------
            self ._warmup_e2e (img_u8 ,self .WARMUPS )

            # ---------------- E2E benchmark ----------------
            med_enc ,med_dec_v ,med_dec_nv ,med_aead_over ,med_aead_verify_only =self ._bench_e2e (img_u8 ,self .RUNS )

            # E2E encryption throughput (u8)
            enc_mbps =_mbps (nbytes_orig ,med_enc )

            # ---------------- phase benchmark ----------------
            pix_times :List [float ]=[]
            blk_times :List [float ]=[]
            msk_times :List [float ]=[]

            for _ in range (int (phase_runs )):
                nonce64 =secrets .randbits (64 )
                pp =self ._sample_phase_params (core )

                t_pix ,c2_u8 =self ._time_pixel_stage (core ,img_u8 ,pp .S ,pp .Kf ,pp .Kr )
                t_blk ,mixed_field =self ._time_block_mix (core ,c2_u8 ,pp .gk_vec ,Hp ,Wp )
                t_msk ,_cblk =self ._time_stream_mask (core ,mixed_field ,pp .key32 ,int (nonce64 ))

                pix_times .append (t_pix )
                blk_times .append (t_blk )
                msk_times .append (t_msk )

            med_pix =_median (pix_times )
            med_blk =_median (blk_times )
            med_msk =_median (msk_times )

            # E2E - (core phases) ≈ overhead (KEM/AEAD/packaging + scheduler jitter)
            overhead_s =max (med_enc -(med_pix +med_blk +med_msk ),0.0 )

            # Phase throughput
            pix_mbps =_mbps (nbytes_orig ,med_pix )
            blk_mbps_u8 =_mbps (nbytes_pad_u8 ,med_blk )
            blk_mbps_u16 =_mbps (nbytes_pad_u16 ,med_blk )
            msk_mbps_u8 =_mbps (nbytes_pad_u8 ,med_msk )
            msk_mbps_u16 =_mbps (nbytes_pad_u16 ,med_msk )

            rows .append ({
            "Image":str (name ),
            "Shape":f"{H}x{W}",
            "bs":int (bs ),
            "Hp":int (Hp ),
            "Wp":int (Wp ),
            "Warmups":int (self .WARMUPS ),
            "Runs_E2E":int (self .RUNS ),
            "Runs_Phase":int (phase_runs ),

            "Enc_total_s_med":float (med_enc ),
            "Dec_total_verify_s_med":float (med_dec_v ),
            "Dec_total_noverify_s_med":float (med_dec_nv ),
            # NOTE: (comment removed; repository is English-only)
            "AEAD_verify_only_s_med":float (med_aead_over ),

            # NOTE: (comment removed; repository is English-only)
            "AEAD_verify_only_direct_s_med":float (med_aead_verify_only ),

            "Pixel_stage_s_med":float (med_pix ),
            "Block_mix_s_med":float (med_blk ),
            "Stream_mask_s_med":float (med_msk ),
            "Overhead_s_med":float (overhead_s ),

            "Enc_MBps_u8":float (enc_mbps ),
            "Pixel_MBps_u8":float (pix_mbps ),
            "Block_MBps_u8eq":float (blk_mbps_u8 ),
            "Block_MBps_fieldbytes":float (blk_mbps_u16 ),
            "Mask_MBps_u8eq":float (msk_mbps_u8 ),
            "Mask_MBps_fieldbytes":float (msk_mbps_u16 ),
            })

            self .log (
            f"  {name}: ENC={med_enc*1e3:.2f} ms, DEC(v)={med_dec_v*1e3:.2f} ms, DEC(nv)={med_dec_nv*1e3:.2f} ms | "
            f"PIX={med_pix*1e3:.2f} ms, BLK={med_blk*1e3:.2f} ms, MSK={med_msk*1e3:.2f} ms, OVH={overhead_s*1e3:.2f} ms"
            )

        df =pd .DataFrame (rows )

        # Kaydet
        self .save_results (df ,"performance_breakdown_pf257.xlsx")

        # LaTeX export (varsa)
        if hasattr (self ,"export_latex")and callable (getattr (self ,"export_latex")):
            self .export_latex (
            df [[
            "Image","Shape","bs",
            "Enc_total_s_med","Dec_total_verify_s_med","Dec_total_noverify_s_med",
            "AEAD_verify_only_s_med",
            "Pixel_stage_s_med","Block_mix_s_med","Stream_mask_s_med","Overhead_s_med",
            "Enc_MBps_u8",
            ]],
            caption ="See README.md for usage and details.",
            label ="tab:perf-breakdown-pf257",
            )

        return df 


        # NOTE: (comment removed; repository is English-only)
        # 6: ExperimentMeta(
        #     cls=PF257PerformanceBenchmark,
        # NOTE: (comment removed; repository is English-only)
        #     params={"runs": 10, "warmups": 3},
        # ),
