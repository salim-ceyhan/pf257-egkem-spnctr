"""
Experiment 11 - Baseline comparison.

Compares PF257 against reference baselines (e.g., ChaCha20-12, AES-CTR) in
terms of metrics and timing.
"""
from __future__ import annotations 

from typing import Any ,Dict ,List ,Tuple ,Optional ,Callable 
import sys 
from pathlib import Path 
import time 
import io 
import gzip 

import numpy as np 
import pandas as pd 

# NOTE: (comment removed; repository is English-only)
try :
    from PIL import Image # type: ignore
    HAS_PIL =True 
except Exception :
    HAS_PIL =False 
    Image =None # type: ignore

    # AES (pycryptodome) opsiyonel
try :
    from Crypto .Cipher import AES # type: ignore
    HAS_PYCRYPTODOME =True 
except Exception :
    HAS_PYCRYPTODOME =False 
    AES =None # type: ignore

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import StatisticalMetrics ,CiphertextView 


# =============================================================================
# ChaCha20-12 (pure-Python, parametrik round)
# =============================================================================
def _rotl32 (x :int ,n :int )->int :
    return ((x <<n )&0xFFFFFFFF )|((x &0xFFFFFFFF )>>(32 -n ))

def _u32le (b :bytes )->int :
    return int .from_bytes (b ,"little",signed =False )

def _chacha_block (key32 :bytes ,counter :int ,nonce12 :bytes ,*,double_rounds :int )->bytes :
    """
    ChaCha block function.
    - double_rounds=10 -> ChaCha20 (20 rounds)
    - double_rounds=6  -> ChaCha12 (12 rounds)
    """
    if len (key32 )!=32 :
        raise ValueError ("ChaCha key must be 32 bytes.")
    if len (nonce12 )!=12 :
        raise ValueError ("ChaCha nonce must be 12 bytes.")

    const =b"expand 32-byte k"
    state =[
    _u32le (const [0 :4 ]),_u32le (const [4 :8 ]),_u32le (const [8 :12 ]),_u32le (const [12 :16 ]),
    _u32le (key32 [0 :4 ]),_u32le (key32 [4 :8 ]),_u32le (key32 [8 :12 ]),_u32le (key32 [12 :16 ]),
    _u32le (key32 [16 :20 ]),_u32le (key32 [20 :24 ]),_u32le (key32 [24 :28 ]),_u32le (key32 [28 :32 ]),
    counter &0xFFFFFFFF ,
    _u32le (nonce12 [0 :4 ]),_u32le (nonce12 [4 :8 ]),_u32le (nonce12 [8 :12 ]),
    ]
    x =state .copy ()

    def qr (a :int ,b :int ,c :int ,d :int )->None :
        x [a ]=(x [a ]+x [b ])&0xFFFFFFFF ;x [d ]^=x [a ];x [d ]=_rotl32 (x [d ],16 )
        x [c ]=(x [c ]+x [d ])&0xFFFFFFFF ;x [b ]^=x [c ];x [b ]=_rotl32 (x [b ],12 )
        x [a ]=(x [a ]+x [b ])&0xFFFFFFFF ;x [d ]^=x [a ];x [d ]=_rotl32 (x [d ],8 )
        x [c ]=(x [c ]+x [d ])&0xFFFFFFFF ;x [b ]^=x [c ];x [b ]=_rotl32 (x [b ],7 )

    for _ in range (int (double_rounds )):
    # column rounds
        qr (0 ,4 ,8 ,12 );qr (1 ,5 ,9 ,13 );qr (2 ,6 ,10 ,14 );qr (3 ,7 ,11 ,15 )
        # diagonal rounds
        qr (0 ,5 ,10 ,15 );qr (1 ,6 ,11 ,12 );qr (2 ,7 ,8 ,13 );qr (3 ,4 ,9 ,14 )

    out =[(x [i ]+state [i ])&0xFFFFFFFF for i in range (16 )]
    return b"".join (w .to_bytes (4 ,"little")for w in out )

def chacha_stream (key32 :bytes ,nonce12 :bytes ,nbytes :int ,*,counter0 :int =0 ,rounds :int =12 )->bytes :
    """
    ChaCha stream generator (RFC8439 layout).
    rounds: 12 veya 20 vb.
    """
    if rounds %2 !=0 :
        raise ValueError ("ChaCha rounds must be even (e.g., 12, 20).")
    double_rounds =rounds //2 

    out =bytearray ()
    counter =counter0 &0xFFFFFFFF 
    while len (out )<int (nbytes ):
        out .extend (_chacha_block (key32 ,counter ,nonce12 ,double_rounds =double_rounds ))
        counter =(counter +1 )&0xFFFFFFFF 
    return bytes (out [:int (nbytes )])


    # =============================================================================
    # Utilities
    # =============================================================================
def _rng (seed :int )->np .random .Generator :
    return np .random .default_rng (int (seed )&0xFFFFFFFF )

def _median (xs :List [float ])->float :
    return float (np .median (np .asarray (xs ,dtype =np .float64 )))

def _mbps (nbytes :int ,seconds :float )->float :
    return float ((float (nbytes )/(1024.0 *1024.0 ))/max (float (seconds ),1e-12 ))

def _png_gzip_ratio_u8 (arr_u8 :np .ndarray )->Tuple [float ,float ]:
    a =np .asarray (arr_u8 ,dtype =np .uint8 ,order ="C")
    raw_bytes =a .tobytes (order ="C")
    raw_size =max (1 ,len (raw_bytes ))

    # PNG ratio (optional)
    if HAS_PIL :
        try :
            buf =io .BytesIO ()
            Image .fromarray (a ,mode ="L").save (buf ,format ="PNG",optimize =True )# type: ignore[arg-type]
            png_ratio =len (buf .getvalue ())/raw_size 
        except Exception :
            png_ratio =float ("nan")
    else :
        png_ratio =float ("nan")

        # GZIP ratio
    try :
        gz_bytes =gzip .compress (raw_bytes ,compresslevel =9 )
        gzip_ratio =len (gz_bytes )/raw_size 
    except Exception :
        gzip_ratio =float ("nan")

    return float (png_ratio ),float (gzip_ratio )

def _stats_2d (img2d :np .ndarray ,*,alphabet :int )->Tuple [float ,float ,float ,Dict [str ,float ]]:
    """See README.md for usage and details."""
    Hx =float (StatisticalMetrics .entropy (img2d ,alphabet =int (alphabet )))
    chi2 ,p =StatisticalMetrics .chi_square (img2d ,alphabet =int (alphabet ))
    corr =StatisticalMetrics .neighbor_correlations (img2d )
    return Hx ,float (chi2 ),float (p ),{
    "rho_h":float (corr ["rho_h"]),
    "rho_v":float (corr ["rho_v"]),
    "rho_d":float (corr ["rho_d"]),
    "rho_a":float (corr ["rho_a"]),
    }

def _as_gray_u8 (img :np .ndarray )->np .ndarray :
    x =np .asarray (img )
    if x .ndim !=2 :
        raise ValueError ("See README.md for usage and details.")
    if x .dtype !=np .uint8 :
    # NOTE: (comment removed; repository is English-only)
        x =np .clip (np .rint (x .astype (np .float64 )),0 ,255 ).astype (np .uint8 )
    return np .asarray (x ,dtype =np .uint8 ,order ="C")

def _nonce63 (rng :np .random .Generator )->int :
# NOTE: (comment removed; repository is English-only)
    return int (rng .integers (0 ,2 **63 -1 ,dtype =np .int64 ))


    # =============================================================================
    # Experiment 11
    # =============================================================================
class PF257PerformanceAndMetricsComparison (BaseExperiment ):
    """Baseline comparison against reference ciphers (Experiment 11)."""

    def __init__ (
    self ,
    config ,
    cipher ,
    *,
    runs :int =10 ,
    warmups :int =3 ,
    enable_aes :Optional [bool ]=None ,
    chacha_rounds :int =12 ,
    **_params :Any ,
    )->None :
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="(PF257) Full Comparison: Ours(Z257 metrics) vs ChaCha/AES (U8 metrics)",
        )

        self .RUNS =int (runs )
        self .WARMUPS =int (warmups )
        self .CHACHA_ROUNDS =int (chacha_rounds )

        self .use_aes =True if enable_aes is None else bool (enable_aes )
        if not HAS_PYCRYPTODOME :
            self .use_aes =False 

            # ------------------------ baseline enc/dec ------------------------
    @staticmethod 
    def _encdec_xor (img_u8 :np .ndarray ,ks_bytes :bytes )->np .ndarray :
        H ,W =img_u8 .shape 
        ks_arr =np .frombuffer (ks_bytes ,dtype =np .uint8 ).reshape (H ,W )
        return np .bitwise_xor (img_u8 ,ks_arr ).astype (np .uint8 ,copy =False )

    def _enc_chacha (self ,img_u8 :np .ndarray ,key32 :bytes ,nonce12 :bytes )->np .ndarray :
        H ,W =img_u8 .shape 
        ks =chacha_stream (key32 ,nonce12 ,H *W ,counter0 =0 ,rounds =self .CHACHA_ROUNDS )
        return self ._encdec_xor (img_u8 ,ks )

    def _dec_chacha (self ,ct_u8 :np .ndarray ,key32 :bytes ,nonce12 :bytes )->np .ndarray :
    # XOR simetrisi
        return self ._enc_chacha (ct_u8 ,key32 ,nonce12 )

    @staticmethod 
    def _enc_aes_ctr (img_u8 :np .ndarray ,key32 :bytes ,nonce8 :bytes )->np .ndarray :
        if not HAS_PYCRYPTODOME :
            raise RuntimeError ("See README.md for usage and details.")
        raw =img_u8 .tobytes (order ="C")
        c =AES .new (key32 ,AES .MODE_CTR ,nonce =nonce8 )# type: ignore
        ct =c .encrypt (raw )
        return np .frombuffer (ct ,dtype =np .uint8 ).reshape (img_u8 .shape )

    @staticmethod 
    def _dec_aes_ctr (ct_u8 :np .ndarray ,key32 :bytes ,nonce8 :bytes )->np .ndarray :
    # NOTE: (comment removed; repository is English-only)
        if not HAS_PYCRYPTODOME :
            raise RuntimeError ("See README.md for usage and details.")
        raw =ct_u8 .tobytes (order ="C")
        c =AES .new (key32 ,AES .MODE_CTR ,nonce =nonce8 )# type: ignore
        pt =c .decrypt (raw )
        return np .frombuffer (pt ,dtype =np .uint8 ).reshape (ct_u8 .shape )

        # ------------------------------- run ------------------------------
    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log (
        f"Experiment 11 started... "
        f"(ChaCha rounds={self.CHACHA_ROUNDS}, AES-CTR={'ON' if self.use_aes else 'OFF'})"
        )

        base_seed =int (getattr (self .config ,"seed",0 ))
        rng =_rng (base_seed )

        rows :List [Dict [str ,Any ]]=[]

        for img_name ,img in images .items ():
            self .log (f"Processing {img_name} ...")

            img_u8 =_as_gray_u8 (img )
            H ,W =img_u8 .shape 
            nbytes =int (H *W )
            shape_str =f"{int(H)}x{int(W)}"

            # =========================================================
            # Ours (PF257) — warmup
            # =========================================================
            for _ in range (self .WARMUPS ):
                nonce =_nonce63 (rng )
                pkt ,_ =self .cipher .encrypt (img_u8 ,nonce =nonce )
                _ =self .cipher .decrypt (pkt ,verify =True )

                # =========================================================
                # Ours (PF257) — timing (median)
                # =========================================================
            t_enc :List [float ]=[]
            t_dec :List [float ]=[]
            C_ours_u8 :Optional [np .ndarray ]=None 
            pkt_last :Optional [dict ]=None 

            for _ in range (self .RUNS ):
                nonce =_nonce63 (rng )

                t0 =time .perf_counter_ns ()
                pkt ,C_ours_u8 =self .cipher .encrypt (img_u8 ,nonce =nonce )
                t1 =time .perf_counter_ns ()
                _ =self .cipher .decrypt (pkt ,verify =True )
                t2 =time .perf_counter_ns ()

                t_enc .append ((t1 -t0 )*1e-9 )
                t_dec .append ((t2 -t1 )*1e-9 )
                pkt_last =pkt 

            med_enc =_median (t_enc )
            med_dec =_median (t_dec )

            if C_ours_u8 is None or pkt_last is None :
                raise RuntimeError ("See README.md for usage and details.")

                # =========================================================
                # Ours — metrics on Z257 if possible (blocks -> field2d)
                # =========================================================
            alphabet_used =256 
            blocks =pkt_last .get ("blocks",None )

            if blocks is not None :
                b =np .asarray (blocks )
                if b .ndim ==4 :
                    field2d =CiphertextView .blocks4d_to_field2d (b .astype (np .uint16 ,copy =False ))
                    Ho ,Chio ,Po ,Ro =_stats_2d (field2d ,alphabet =257 )
                    alphabet_used =257 
                else :
                    Ho ,Chio ,Po ,Ro =_stats_2d (C_ours_u8 ,alphabet =256 )
            else :
                Ho ,Chio ,Po ,Ro =_stats_2d (C_ours_u8 ,alphabet =256 )

            pngo ,gzo =_png_gzip_ratio_u8 (C_ours_u8 )

            rows .append ({
            "Image":img_name ,
            "Shape":shape_str ,
            "Scheme":"PF257_EGKEM_SPNCTR (Ours)",
            "Alphabet":int (alphabet_used ),

            "Enc_s":med_enc ,
            "Enc_MBps":_mbps (nbytes ,med_enc ),
            "Dec_s":med_dec ,
            "Dec_MBps":_mbps (nbytes ,med_dec ),

            "Entropy":Ho ,
            "Chi2":Chio ,
            "p":Po ,
            "rho_h":Ro ["rho_h"],"rho_v":Ro ["rho_v"],"rho_d":Ro ["rho_d"],"rho_a":Ro ["rho_a"],

            "PNG_ratio":pngo ,
            "GZIP_ratio":gzo ,
            "Runs":self .RUNS ,
            })

            # =========================================================
            # ChaCha20-12 — warmup + timing (ENC/DEC) + metrics
            # =========================================================
            key_ch =rng .integers (0 ,256 ,32 ,dtype =np .uint8 ).tobytes ()
            for _ in range (self .WARMUPS ):
                nonce12 =rng .integers (0 ,256 ,12 ,dtype =np .uint8 ).tobytes ()
                _ =self ._enc_chacha (img_u8 ,key_ch ,nonce12 )

            t_ch_enc :List [float ]=[]
            t_ch_dec :List [float ]=[]
            Cch :Optional [np .ndarray ]=None 

            # NOTE: (comment removed; repository is English-only)
            nonce12_last :Optional [bytes ]=None 
            for _ in range (self .RUNS ):
                nonce12 =rng .integers (0 ,256 ,12 ,dtype =np .uint8 ).tobytes ()
                nonce12_last =nonce12 

                t0 =time .perf_counter_ns ()
                Cch =self ._enc_chacha (img_u8 ,key_ch ,nonce12 )
                t1 =time .perf_counter_ns ()
                _ =self ._dec_chacha (Cch ,key_ch ,nonce12 )
                t2 =time .perf_counter_ns ()

                t_ch_enc .append ((t1 -t0 )*1e-9 )
                t_ch_dec .append ((t2 -t1 )*1e-9 )

            if Cch is None or nonce12_last is None :
                raise RuntimeError ("See README.md for usage and details.")

            med_ch_enc =_median (t_ch_enc )
            med_ch_dec =_median (t_ch_dec )

            Hc ,Chic ,Pc ,Rc =_stats_2d (Cch ,alphabet =256 )
            pngc ,gzc =_png_gzip_ratio_u8 (Cch )

            rows .append ({
            "Image":img_name ,
            "Shape":shape_str ,
            "Scheme":f"ChaCha20-{self.CHACHA_ROUNDS}",
            "Alphabet":256 ,

            "Enc_s":med_ch_enc ,
            "Enc_MBps":_mbps (nbytes ,med_ch_enc ),
            "Dec_s":med_ch_dec ,
            "Dec_MBps":_mbps (nbytes ,med_ch_dec ),

            "Entropy":Hc ,
            "Chi2":Chic ,
            "p":Pc ,
            "rho_h":Rc ["rho_h"],"rho_v":Rc ["rho_v"],"rho_d":Rc ["rho_d"],"rho_a":Rc ["rho_a"],

            "PNG_ratio":pngc ,
            "GZIP_ratio":gzc ,
            "Runs":self .RUNS ,
            })

            # =========================================================
            # AES-CTR-256 — warmup + timing (ENC/DEC) + metrics
            # =========================================================
            if self .use_aes :
                key_aes =rng .integers (0 ,256 ,32 ,dtype =np .uint8 ).tobytes ()

                for _ in range (self .WARMUPS ):
                    nonce8 =rng .integers (0 ,256 ,8 ,dtype =np .uint8 ).tobytes ()
                    _ =self ._enc_aes_ctr (img_u8 ,key_aes ,nonce8 )

                t_aes_enc :List [float ]=[]
                t_aes_dec :List [float ]=[]
                Caes :Optional [np .ndarray ]=None 
                nonce8_last :Optional [bytes ]=None 

                for _ in range (self .RUNS ):
                    nonce8 =rng .integers (0 ,256 ,8 ,dtype =np .uint8 ).tobytes ()
                    nonce8_last =nonce8 

                    t0 =time .perf_counter_ns ()
                    Caes =self ._enc_aes_ctr (img_u8 ,key_aes ,nonce8 )
                    t1 =time .perf_counter_ns ()
                    _ =self ._dec_aes_ctr (Caes ,key_aes ,nonce8 )
                    t2 =time .perf_counter_ns ()

                    t_aes_enc .append ((t1 -t0 )*1e-9 )
                    t_aes_dec .append ((t2 -t1 )*1e-9 )

                if Caes is None or nonce8_last is None :
                    raise RuntimeError ("See README.md for usage and details.")

                med_aes_enc =_median (t_aes_enc )
                med_aes_dec =_median (t_aes_dec )

                Ha ,Chia ,Pa ,Ra =_stats_2d (Caes ,alphabet =256 )
                pnga ,gza =_png_gzip_ratio_u8 (Caes )

                rows .append ({
                "Image":img_name ,
                "Shape":shape_str ,
                "Scheme":"AES-CTR-256",
                "Alphabet":256 ,

                "Enc_s":med_aes_enc ,
                "Enc_MBps":_mbps (nbytes ,med_aes_enc ),
                "Dec_s":med_aes_dec ,
                "Dec_MBps":_mbps (nbytes ,med_aes_dec ),

                "Entropy":Ha ,
                "Chi2":Chia ,
                "p":Pa ,
                "rho_h":Ra ["rho_h"],"rho_v":Ra ["rho_v"],"rho_d":Ra ["rho_d"],"rho_a":Ra ["rho_a"],

                "PNG_ratio":pnga ,
                "GZIP_ratio":gza ,
                "Runs":self .RUNS ,
                })

                # =============================================================
                # LONG table
                # =============================================================
        df_long =pd .DataFrame (rows ,columns =[
        "Image","Shape","Scheme","Alphabet",
        "Enc_s","Enc_MBps","Dec_s","Dec_MBps",
        "Entropy","Chi2","p","rho_h","rho_v","rho_d","rho_a",
        "PNG_ratio","GZIP_ratio","Runs"
        ])
        self .save_results (df_long ,"perf_metrics_comparison_long.xlsx")

        # =============================================================
        # WIDE table (MB/s summary)
        # =============================================================
        df_pivot =df_long .pivot (
        index =["Image","Shape"],
        columns ="Scheme",
        values =["Enc_MBps","Dec_MBps"],
        )
        df_pivot .columns =["_".join (col ).strip ()for col in df_pivot .columns .values ]
        df_wide =df_pivot .reset_index ()

        # NOTE: (comment removed; repository is English-only)
        chacha_name =f"ChaCha20-{self.CHACHA_ROUNDS}"

        colmap ={
        "Enc_MBps_PF257_EGKEM_SPNCTR (Ours)":"Ours_Enc_MBps",
        "Dec_MBps_PF257_EGKEM_SPNCTR (Ours)":"Ours_Dec_MBps",
        f"Enc_MBps_{chacha_name}":"ChaCha_Enc_MBps",
        f"Dec_MBps_{chacha_name}":"ChaCha_Dec_MBps",
        "Enc_MBps_AES-CTR-256":"AES_Enc_MBps",
        "Dec_MBps_AES-CTR-256":"AES_Dec_MBps",
        }
        df_wide =df_wide .rename (columns =colmap )

        ordered =[
        "Image","Shape",
        "Ours_Enc_MBps","Ours_Dec_MBps",
        "ChaCha_Enc_MBps","ChaCha_Dec_MBps",
        "AES_Enc_MBps","AES_Dec_MBps",
        ]
        df_wide_final =df_wide [[c for c in ordered if c in df_wide .columns ]]
        self .save_results (df_wide_final ,"perf_comparison_wide.xlsx")

        # =============================================================
        # LaTeX export (wide)
        # =============================================================
        try :
            self .export_latex (
            df_wide_final ,
            caption ="Performance comparison: PF257 full pipeline vs. ChaCha20 and AES-CTR (MB/s, median of runs).",
            label ="tab:perf-compare",
            )
        except Exception :
            pass 

        self .log ("Experiment 11 completed.")
        return df_long 
