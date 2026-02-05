# exp_13_pf257_timing_leakage.py
# ---------------------------------------------------------------------
# Experiment 13 — Timing Leakage Sanity (content–time correlation)
# NOTE: (comment removed; repository is English-only)
# NOTE: (comment removed; repository is English-only)
# NOTE: (comment removed; repository is English-only)
# NOTE: (comment removed; repository is English-only)
# NOTE: (comment removed; repository is English-only)
#   - timing_sanity_raw.xlsx
#   - timing_sanity_pattern_summary.xlsx
#   - timing_sanity_correlations.xlsx
# ---------------------------------------------------------------------
from __future__ import annotations 

import time 
from typing import Any ,Dict ,List 
import sys 
from pathlib import Path 
import secrets 

import numpy as np 
import pandas as pd 

from pf257_experiment_base import BaseExperiment 


class PF257TimingLeakageSanity (BaseExperiment ):
    """Timing leakage sanity checks (Experiment 13)."""

    def __init__ (
    self ,
    config ,
    cipher ,
    *,
    repeats :int =20 ,
    trials :int =50 ,
    warmups :int =5 ,
    shuffle_patterns :bool =True ,
    **_params :Any ,
    )->None :
    # --- dinamik isim ---
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")
        # -------------------

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="(PF257) Timing Leakage Sanity (content–time correlation)",
        )

        self .REPEATS =int (repeats )
        self .TRIALS =int (trials )
        self .WARMUPS =int (warmups )
        self .SHUFFLE =bool (shuffle_patterns )

    @staticmethod 
    def _rng (seed :int )->np .random .Generator :
        return np .random .default_rng (int (seed )&0xFFFFFFFF )

    @staticmethod 
    def _patterns (H :int ,W :int ,rng :np .random .Generator )->Dict [str ,np .ndarray ]:
        z =np .zeros ((H ,W ),dtype =np .uint8 )
        o =np .full ((H ,W ),255 ,dtype =np .uint8 )

        chk =(np .indices ((H ,W )).sum (axis =0 )&1 ).astype (np .uint8 )*255 

        stripes =(np .arange (W )%8 <4 ).astype (np .uint8 )*255 
        stripes =np .tile (stripes ,(H ,1 )).astype (np .uint8 )

        rnd =rng .integers (0 ,256 ,size =(H ,W ),dtype =np .uint8 )

        return {"zeros":z ,"ones":o ,"checker":chk ,"stripes":stripes ,"random":rnd }

    @staticmethod 
    def _runs_h (mat_u8 :np .ndarray )->float :
        H ,W =mat_u8 .shape 
        if W <=1 :
            return 0.0 
            # NOTE: (comment removed; repository is English-only)
        return float (np .mean (np .sum (mat_u8 [:,1 :]!=mat_u8 [:,:-1 ],axis =1 )))

    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Timing leakage sanity started...")

        rows :List [Dict [str ,Any ]]=[]
        seed =int (getattr (self .config ,"seed",0 ))
        rng_global =self ._rng (seed )

        for img_name ,img in images .items ():
            img_u8 =np .asarray (img ,dtype =np .uint8 ,order ="C")
            if img_u8 .ndim !=2 :
                raise ValueError ("See README.md for usage and details.")

            H ,W =img_u8 .shape 
            shape_str =f"{int(H)}x{int(W)}"

            # NOTE: (comment removed; repository is English-only)
            rng =self ._rng (seed ^(hash (img_name )&0xFFFFFFFF ))
            pats =self ._patterns (H ,W ,rng )

            labels =list (pats .keys ())

            for rep in range (self .REPEATS ):
            # NOTE: (comment removed; repository is English-only)
                if self .SHUFFLE :
                    rng_global .shuffle (labels )

                for label in labels :
                    mat =np .asarray (pats [label ],dtype =np .uint8 ,order ="C")

                    # NOTE: (comment removed; repository is English-only)
                    if self .WARMUPS >0 :
                        warm_nonces =[secrets .randbits (63 )for _ in range (self .WARMUPS )]
                        for wn in warm_nonces :
                            _ =self .cipher .encrypt (mat ,nonce =wn )

                            # NOTE: (comment removed; repository is English-only)
                    nonces =[secrets .randbits (63 )for _ in range (self .TRIALS )]

                    t =np .empty (self .TRIALS ,dtype =np .float64 )

                    for i in range (self .TRIALS ):
                        t0 =time .perf_counter_ns ()
                        _ =self .cipher .encrypt (mat ,nonce =nonces [i ])
                        t1 =time .perf_counter_ns ()
                        t [i ]=(t1 -t0 )*1e-9 

                    t_med =float (np .median (t ))

                    rows .append ({
                    "Image":img_name ,
                    "Shape":shape_str ,
                    "Repeat":int (rep ),
                    "Pattern":str (label ),
                    "Enc_median_s":float (t_med ),
                    "byte_mean":float (np .mean (mat )),
                    "byte_std":float (np .std (mat )),
                    "runs_h":float (self ._runs_h (mat )),
                    })

        df_raw =pd .DataFrame (rows ,columns =[
        "Image","Shape","Repeat","Pattern",
        "Enc_median_s","byte_mean","byte_std","runs_h"
        ])
        self .save_results (df_raw ,"timing_sanity_raw.xlsx")

        # ------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------
        df_summary =(
        df_raw 
        .groupby (["Image","Shape","Pattern"],as_index =False )
        .agg (
        Enc_median_s_median =("Enc_median_s","median"),
        Enc_median_s_mean =("Enc_median_s","mean"),
        Enc_median_s_std =("Enc_median_s","std"),
        n =("Enc_median_s","count"),
        )
        )
        self .save_results (df_summary ,"timing_sanity_pattern_summary.xlsx")

        # ------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------
        corr_rows :List [Dict [str ,Any ]]=[]
        for (img_name ,shape_str ),g in df_raw .groupby (["Image","Shape"],as_index =False ):
            corr_cols =["Enc_median_s","byte_mean","byte_std","runs_h"]
            corr =g [corr_cols ].corr (method ="pearson")

            # NOTE: (comment removed; repository is English-only)
            v_mean :Any =corr .loc ["Enc_median_s","byte_mean"]
            v_std :Any =corr .loc ["Enc_median_s","byte_std"]
            v_runs :Any =corr .loc ["Enc_median_s","runs_h"]

            r_mean =float (v_mean )
            r_std =float (v_std )
            r_runs =float (v_runs )

            corr_rows .append ({
            "Image":img_name ,
            "Shape":shape_str ,
            "Corr_Time_vs_Mean":r_mean ,
            "Corr_Time_vs_Std":r_std ,
            "Corr_Time_vs_RunsH":r_runs ,
            })

            self .log (
            f"{img_name} ({shape_str}) | "
            f"Corr(Time,Mean)={r_mean:+.4f}, "
            f"Corr(Time,Std)={r_std:+.4f}, "
            f"Corr(Time,RunsH)={r_runs:+.4f}"
            )

        df_corr =pd .DataFrame (corr_rows ,columns =[
        "Image","Shape",
        "Corr_Time_vs_Mean","Corr_Time_vs_Std","Corr_Time_vs_RunsH"
        ])
        self .save_results (df_corr ,"timing_sanity_correlations.xlsx")

        return df_raw 
