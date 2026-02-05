# exp_01_pf257_stage_metrics_paper.py
# ---------------------------------------------------------------------
# PF257-EGKEM-SPNCTR — Stage-by-stage pipeline analysis (Paper-ready)
# Outputs:
#   - stage_metrics_pf257_long.xlsx
#   - stage_metrics_pf257_wide.xlsx
#   - stage_metrics_pf257_sanity.xlsx
#   - stage_metrics_pf257_stage_summary.xlsx
#   - stage_metrics_pf257_stage_summary_subtables.tex   (paper-ready Table*)
# Figures:
#   - fig_stage/*  (P, PF_u8, C_u8 images)
#   - fig_corr/*   (neighbor correlation scatter triplet)
#   - fig_hist/*   (histograms on alphabets 256/257)
# ---------------------------------------------------------------------
from __future__ import annotations 

from typing import Any ,Dict ,Tuple ,List 
import sys 
from pathlib import Path 
from math import log10 
import hashlib 
from cryptography .hazmat .primitives .asymmetric import x25519 

import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 
import matplotlib as mpl 

mpl .rcParams .update ({
"pdf.fonttype":42 ,# NOTE: (comment removed; repository is English-only)
"ps.fonttype":42 ,
"savefig.bbox":"tight",
})


try :
    from PIL import Image 
    HAS_PIL =True 
except Exception :
    HAS_PIL =False 
    Image =None # type: ignore

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import StatisticalMetrics ,CiphertextView 


class PF257StageTripletMetrics (BaseExperiment ):
    """Stage-wise P/PF/C metrics (entropy, correlations) (Experiment 01)."""

    def __init__ (self ,config ,cipher ,*,scatter_samples :int =4000 ,**_params :Any ):
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="PF257 stage-by-stage analysis (P—PF—C) — paper-ready outputs",
        )

        self .scatter_samples =int (scatter_samples )

        # ---------------------------
        # Image helper (color->gray)
        # ---------------------------
    @staticmethod 
    def _to_gray_u8 (img :np .ndarray )->np .ndarray :
        """See README.md for usage and details."""
        x =np .asarray (img )
        if x .ndim ==2 :
            return np .asarray (x ,dtype =np .uint8 ,order ="C")
        if x .ndim ==3 and x .shape [-1 ]in (3 ,4 ):
            g =np .mean (x [...,:3 ],axis =2 )
            return np .asarray (np .clip (np .rint (g ),0 ,255 ),dtype =np .uint8 ,order ="C")
        raise ValueError (f"Unsupported image shape: {x.shape}")

        # ---------------------------
        # deterministic nonce (exp01/exp02 common)
        # ---------------------------
    def _nonce64 (self ,img_name :str ,*,domain_tag :str ="EXP01")->int :
        seed =int (getattr (self .config ,"seed",0 ))
        h =hashlib .blake2b (f"PF257_NONCE|{domain_tag}|{seed}|{img_name}".encode ("utf-8"),digest_size =8 ).digest ()
        nonce =int .from_bytes (h ,"big",signed =False )&((1 <<63 )-1 )
        return 1 if nonce ==0 else int (nonce )

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _get_pf257_core_ctx (self )->tuple [Any ,x25519 .X25519PrivateKey ,bytes ]:
        cw =self .cipher 
        adapter =getattr (cw ,"cipher",None )
        if adapter is None :
            raise AttributeError ("See README.md for usage and details.")

        core =getattr (adapter ,"core",None )
        bob_priv =getattr (adapter ,"bob_priv_x",None )# X25519PrivateKey
        bob_pub =getattr (adapter ,"bob_pub_h",None )# bytes (raw pub)

        if core is None or bob_priv is None or bob_pub is None :
            raise AttributeError ("See README.md for usage and details.")

        if not isinstance (bob_priv ,x25519 .X25519PrivateKey ):
            raise TypeError ("See README.md for usage and details.")
        if not isinstance (bob_pub ,(bytes ,bytearray ))or len (bob_pub )!=32 :
            raise TypeError ("See README.md for usage and details.")

        return core ,bob_priv ,bytes (bob_pub )

    @staticmethod 
    def _crop (img :np .ndarray ,shape :Tuple [int ,int ])->np .ndarray :
        H ,W =int (shape [0 ]),int (shape [1 ])
        return img [:H ,:W ]

    @staticmethod 
    def _safe_log10_p (p :float )->float :
        return float (log10 (max (float (p ),1e-300 )))

        # ------------------------------------------------------------------
        # Stage metrikleri
        # ------------------------------------------------------------------
    @staticmethod 
    def _metrics_u8 (u8 :np .ndarray )->Dict [str ,float ]:
        u8 =np .asarray (u8 ,dtype =np .uint8 ,order ="C")
        H =float (StatisticalMetrics .entropy (u8 ,alphabet =256 ))
        corr =StatisticalMetrics .neighbor_correlations (u8 )
        return {
        "Entropy_256":H ,
        "rho_h_256":float (corr ["rho_h"]),
        "rho_v_256":float (corr ["rho_v"]),
        "rho_d_256":float (corr ["rho_d"]),
        "rho_a_256":float (corr ["rho_a"]),
        }

    @staticmethod 
    def _metrics_field (field :np .ndarray )->Dict [str ,float ]:
        f =np .asarray (field ,dtype =np .uint16 ,order ="C")
        H =float (StatisticalMetrics .entropy (f ,alphabet =257 ))
        chi2 ,p =StatisticalMetrics .chi_square (f ,alphabet =257 )
        corr =StatisticalMetrics .neighbor_correlations (f )
        return {
        "Entropy_257":H ,
        "Chi2_257":float (chi2 ),
        "p_257":float (p ),
        "log10p_257":PF257StageTripletMetrics ._safe_log10_p (float (p )),
        "rho_h_257":float (corr ["rho_h"]),
        "rho_v_257":float (corr ["rho_v"]),
        "rho_d_257":float (corr ["rho_d"]),
        "rho_a_257":float (corr ["rho_a"]),
        }

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _extract_stages_pf257 (
    self ,
    core :Any ,
    img_u8 :np .ndarray ,
    packet :Dict [str ,Any ],
    bob_priv :Any ,
    bob_pub_raw :bytes ,
    )->Dict [str ,Dict [str ,Any ]]:

        img_u8 =np .asarray (img_u8 ,dtype =np .uint8 ,order ="C")
        H ,W =img_u8 .shape 

        orig_shape =tuple (packet .get ("orig_shape",(H ,W )))

        blocks =packet .get ("blocks",None )
        if blocks is None :
            raise KeyError ("packet['blocks'] not found.")

        C_field_pad =CiphertextView .blocks4d_to_field2d (np .asarray (blocks ,dtype =np .uint16 ))
        Hp ,Wp =C_field_pad .shape 

        kem_epk =packet ["kem_epk"]# 32-byte
        shared =core ._kem_decapsulate (bob_priv ,bytes (kem_epk ))
        ctx =core ._ctx (bob_pub_raw =bytes (bob_pub_raw ),kem_epk_raw =bytes (kem_epk ))

        plain =core ._unwrap_params (shared ,packet ["salt"],ctx ,packet ["enc_params"])
        _ ,_ ,_ ,_ ,key32 ,_ =core ._unpack_params (plain )

        nonce =int (packet ["nonce"])
        stream =core ._field_stream (key32 ,Hp *Wp ,nonce ).reshape (Hp ,Wp ).astype (np .int32 )
        PF_field_pad =((C_field_pad .astype (np .int32 )-stream )%int (core .P_FIELD )).astype (np .uint16 )

        PF_field =self ._crop (PF_field_pad ,orig_shape )
        C_field =self ._crop (C_field_pad ,orig_shape )

        PF_u8 =CiphertextView .field2d_to_u8 (PF_field )
        C_u8 =CiphertextView .field2d_to_u8 (C_field )

        return {
        "P":{"label":"P","u8":img_u8 ,"field":None },
        "PF":{"label":"PF","u8":PF_u8 ,"field":PF_field },
        "C":{"label":"C","u8":C_u8 ,"field":C_field },
        }

        # ------------------------------------------------------------------
        # Figure helpers
        # ------------------------------------------------------------------
    @staticmethod 
    def _save_gray_u8 (u8 :np .ndarray ,path :Path )->None :
        arr =np .asarray (u8 ,dtype =np .uint8 ,order ="C")
        if HAS_PIL :
            Image .fromarray (arr ).save (path ,dpi =(300 ,300 ))# type: ignore # dpi meta
        else :
            plt .imsave (path ,arr ,cmap ="gray",vmin =0 ,vmax =255 )


    def _save_stage_images (self ,img_name :str ,stages :Dict [str ,Dict [str ,Any ]],out_dir :Path )->None :
        base =Path (img_name ).stem 
        for k in ["P","PF","C"]:
            u8 =np .asarray (stages [k ]["u8"],dtype =np .uint8 ,order ="C")
            fn =out_dir /f"{base}_{k.lower()}.png"
            self ._save_gray_u8 (u8 ,fn )

    def _extract_pairs (self ,img :np .ndarray ,direction :str )->Tuple [np .ndarray ,np .ndarray ]:
        img =np .asarray (img ,dtype =np .uint8 ,order ="C")
        if direction =="horizontal":
            x =img [:,:-1 ].ravel ();y =img [:,1 :].ravel ()
        elif direction =="vertical":
            x =img [:-1 ,:].ravel ();y =img [1 :,:].ravel ()
        elif direction =="diagonal":
            x =img [:-1 ,:-1 ].ravel ();y =img [1 :,1 :].ravel ()
        elif direction =="anti-diagonal":
            x =img [:-1 ,1 :].ravel ();y =img [1 :,:-1 ].ravel ()
        else :
            raise ValueError (f"Unknown direction: {direction}")

        n =int (self .scatter_samples )
        if x .size >n :
            seed =int (getattr (self .config ,"seed",0 ))
            rng =np .random .default_rng (seed &0xFFFFFFFF )
            idx =rng .choice (x .size ,size =n ,replace =False )
            x =x [idx ];y =y [idx ]
        return x ,y 

    def _save_correlation_plots_triplet (self ,img_name :str ,stages :Dict [str ,Dict [str ,Any ]],out_dir :Path )->None :
        fig ,axes =plt .subplots (3 ,4 ,figsize =(9.0 ,6.0 ))

        stage_list =[("P","P"),("PF","PF_view"),("C","C_view")]
        # NOTE: (comment removed; repository is English-only)
        directions =["horizontal","vertical","diagonal","anti-diagonal"]
        # NOTE: (comment removed; repository is English-only)
        col_titles =[r"$\rho_h$",r"$\rho_v$",r"$\rho_d$",r"$\rho_a$"]

        for row ,(k ,label )in enumerate (stage_list ):
            stage_img =np .asarray (stages [k ]["u8"],dtype =np .uint8 ,order ="C")
            for col ,(direction_key ,col_title )in enumerate (zip (directions ,col_titles )):
                x ,y =self ._extract_pairs (stage_img ,direction_key )
                axes [row ,col ].scatter (x ,y ,s =1 ,alpha =0.25 ,rasterized =True )
                axes [row ,col ].set_xlim ([0 ,255 ]);axes [row ,col ].set_ylim ([0 ,255 ])
                axes [row ,col ].set_aspect ("equal")
                axes [row ,col ].set_title (f"{label} — {col_title}",fontsize =8 )
                axes [row ,col ].set_xlabel ("")
                axes [row ,col ].set_ylabel ("")

        fig .supxlabel ("Neighbor pair value (x)",fontsize =9 )
        fig .supylabel ("Neighbor pair value (y)",fontsize =9 )

        plt .tight_layout ()

        stem =Path (img_name ).stem 

        # NOTE: (comment removed; repository is English-only)
        out_pdf =out_dir /f"{stem}_corr_pf257.pdf"
        plt .savefig (out_pdf )# NOTE: (comment removed; repository is English-only)
        self .log (f"  ✓ Corr plot (PDF): {out_pdf.name}")

        # NOTE: (comment removed; repository is English-only)
        out_png =out_dir /f"{stem}_corr_pf257.png"
        plt .savefig (out_png ,dpi =600 )
        self .log (f"  ✓ Corr plot (PNG 600dpi): {out_png.name}")

        plt .close ()

    def _save_histograms_triplet (self ,img_name :str ,stages :Dict [str ,Dict [str ,Any ]],out_dir :Path )->None :
        base =Path (img_name ).stem 
        fig ,axes =plt .subplots (3 ,1 ,figsize =(8.0 ,6.0 ))

        P_u8 =np .asarray (stages ["P"]["u8"],dtype =np .uint8 ,order ="C").ravel ()
        axes [0 ].hist (P_u8 ,bins =256 ,range =(0 ,256 ))
        axes [0 ].set_title ("P: intensity histogram (256 levels)",fontsize =9 )

        PF_f =np .asarray (stages ["PF"]["field"],dtype =np .uint16 ,order ="C").ravel ()
        axes [1 ].hist (PF_f ,bins =257 ,range =(0 ,257 ))
        axes [1 ].set_title ("PF: symbol histogram in F_257 (257 symbols)",fontsize =9 )

        C_f =np .asarray (stages ["C"]["field"],dtype =np .uint16 ,order ="C").ravel ()
        axes [2 ].hist (C_f ,bins =257 ,range =(0 ,257 ))
        axes [2 ].set_title ("C: symbol histogram in F_257 (257 symbols)",fontsize =9 )
        plt .tight_layout ()
        stem =Path (img_name ).stem 

        out_pdf =out_dir /f"{stem}_hist_pf257.pdf"
        plt .savefig (out_pdf )
        self .log (f"  ✓ Hist (PDF): {out_pdf.name}")

        out_png =out_dir /f"{stem}_hist_pf257.png"
        plt .savefig (out_png ,dpi =600 )
        self .log (f"  ✓ Hist (PNG 600dpi): {out_png.name}")

        plt .close ()


        # ------------------------------------------------------------------
        # LaTeX Table* builder (paper-ready subtables a,b,c)
        # ------------------------------------------------------------------
    @staticmethod 
    def _pm (mean :float ,std :float ,fmt :str =".4f")->str :
        if np .isnan (mean )or np .isnan (std ):
            return "--"
        return f"${mean:{fmt}} \\pm {std:{fmt}}$"

    def _build_paper_table_tex (self ,df_long :pd .DataFrame ,df_sanity :pd .DataFrame )->str :
    # --- Z256 summary (P/PF/C on u8 domain) ---
        def ms (stage :str ,col :str )->Tuple [float ,float ]:
            s =df_long .loc [df_long ["StageKey"]==stage ,col ].astype (float )
            return float (s .mean ()),float (s .std (ddof =0 ))

        stages =["P","PF","C"]
        cols256 =["Entropy_256","rho_h_256","rho_v_256","rho_d_256","rho_a_256"]
        rows_a ={st :{c :self ._pm (*ms (st ,c ))for c in cols256 }for st in stages }

        # --- Z257 summary (PF/C on field domain; P is --) ---
        cols257 =["Entropy_257","log10p_257","rho_h_257","rho_v_257","rho_d_257","rho_a_257"]

        def ms257 (stage :str ,col :str )->Tuple [float ,float ]:
            s =df_long .loc [df_long ["StageKey"]==stage ,col ].astype (float )
            return float (s .mean ()),float (s .std (ddof =0 ))

        rows_b ={}
        for st in stages :
            if st =="P":
                rows_b [st ]={c :"--"for c in cols257 }
            else :
                rows_b [st ]={c :self ._pm (*ms257 (st ,c ))for c in cols257 }

                # --- PF->C per-image table (use df_sanity) ---
        dfc =df_sanity .copy ()
        dfc ["H"]=dfc ["H"].astype (int );dfc ["W"]=dfc ["W"].astype (int )
        dfc ["CR"]=dfc ["ChangeRate_Field"].astype (float )
        dfc ["Delta"]=dfc ["Delta_ChangeRate"].astype (float )

        # latex string
        tex =r"""See README.md for usage and details."""
        for st in stages :
            tex +=(
            f"\\textbf{{{st}}} & {rows_a[st]['Entropy_256']} & {rows_a[st]['rho_h_256']} & "
            f"{rows_a[st]['rho_v_256']} & {rows_a[st]['rho_d_256']} & {rows_a[st]['rho_a_256']} \\\\\n"
            )
        tex +=r"""See README.md for usage and details."""
        for st in stages :
            tex +=(
            f"\\textbf{{{st}}} & {rows_b[st]['Entropy_257']} & {rows_b[st]['log10p_257']} & "
            f"{rows_b[st]['rho_h_257']} & {rows_b[st]['rho_v_257']} & {rows_b[st]['rho_d_257']} & "
            f"{rows_b[st]['rho_a_257']} \\\\\n"
            )
        tex +=r"""See README.md for usage and details."""
        for _ ,r in dfc .iterrows ():
            img =str (r ["Image"])
            tex +=f"{Path(img).stem} & ${int(r['H'])}\\times{int(r['W'])}$ & {float(r['CR']):.9f} & ${float(r['Delta']):+.2e}$ \\\\\n"

        tex +=r"""\bottomrule
\end{tabular}
}
\end{table*}
"""
        return tex 

        # ------------------------------------------------------------------
        # Main
        # ------------------------------------------------------------------
    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Starting PF257 stage-by-stage analysis (paper-ready)...")

        core ,bob_priv ,bob_pub_raw =self ._get_pf257_core_ctx ()

        fig_stage =self .output_dir /"fig_stage"
        fig_corr =self .output_dir /"fig_corr"
        fig_hist =self .output_dir /"fig_hist"
        for d in (fig_stage ,fig_corr ,fig_hist ):
            d .mkdir (parents =True ,exist_ok =True )

        rows_long :List [Dict [str ,Any ]]=[]
        rows_sanity :List [Dict [str ,Any ]]=[]

        for img_name ,img in images .items ():
            self .log (f"Processing {img_name}...")

            P =self ._to_gray_u8 (img )
            H ,W =P .shape 
            shape_str =f"{int(H)}x{int(W)}"

            nonce =self ._nonce64 (img_name ,domain_tag ="EXP01")

            packet ,_ =self .cipher .encrypt (P ,nonce =int (nonce ))
            stages =self ._extract_stages_pf257 (core ,P ,packet ,bob_priv ,bob_pub_raw )

            PF_field =np .asarray (stages ["PF"]["field"],dtype =np .uint16 ,order ="C")
            C_field =np .asarray (stages ["C"]["field"],dtype =np .uint16 ,order ="C")

            change_rate =float (np .mean (PF_field !=C_field ))
            expected =float (256.0 /257.0 )
            delta =float (change_rate -expected )

            rows_sanity .append ({
            "Image":img_name ,
            "H":int (H ),
            "W":int (W ),
            "Nonce64":int (nonce ),
            "ChangeRate_Field":float (change_rate ),
            "Expected_ChangeRate":float (expected ),
            "Delta_ChangeRate":float (delta ),
            })

            for k in ["P","PF","C"]:
                u8 =np .asarray (stages [k ]["u8"],dtype =np .uint8 ,order ="C")
                m256 =self ._metrics_u8 (u8 )

                if k =="P":
                    m257 ={
                    "Entropy_257":float ("nan"),"Chi2_257":float ("nan"),
                    "p_257":float ("nan"),"log10p_257":float ("nan"),
                    "rho_h_257":float ("nan"),"rho_v_257":float ("nan"),
                    "rho_d_257":float ("nan"),"rho_a_257":float ("nan"),
                    }
                else :
                    field =np .asarray (stages [k ]["field"],dtype =np .uint16 ,order ="C")
                    m257 =self ._metrics_field (field )

                rows_long .append ({
                "Image":img_name ,
                "H":int (H ),
                "W":int (W ),
                "Nonce64":int (nonce ),
                "StageKey":k ,
                "Stage":str (stages [k ]["label"]),
                **m256 ,
                **m257 ,
                })

            self ._save_stage_images (img_name ,stages ,fig_stage )
            self ._save_correlation_plots_triplet (img_name ,stages ,fig_corr )
            self ._save_histograms_triplet (img_name ,stages ,fig_hist )

            # SANITY
        df_sanity =pd .DataFrame (rows_sanity )
        self .save_results (df_sanity ,"stage_metrics_pf257_sanity.xlsx")

        # LONG
        df_long =pd .DataFrame (rows_long )
        self .save_results (df_long ,"stage_metrics_pf257_long.xlsx")

        # WIDE (paper-friendly per-image)
        metrics_for_wide =[
        "Entropy_256","rho_h_256","rho_v_256","rho_d_256","rho_a_256",
        "Entropy_257","log10p_257","rho_h_257","rho_v_257","rho_d_257","rho_a_257",
        ]
        df_wide =df_long .pivot (index =["Image","H","W","Nonce64"],columns ="StageKey",values =metrics_for_wide )
        df_wide .columns =[f"{m}_{k}"for (m ,k )in df_wide .columns ]
        df_wide =df_wide .reset_index ()
        self .save_results (df_wide ,"stage_metrics_pf257_wide.xlsx")

        # STAGE SUMMARY (numeric)
        grp =df_long .groupby ("StageKey",as_index =False )
        df_sum =grp .agg ({
        "Entropy_256":["mean","std"],
        "rho_h_256":["mean","std"],
        "rho_v_256":["mean","std"],
        "rho_d_256":["mean","std"],
        "rho_a_256":["mean","std"],
        "Entropy_257":["mean","std"],
        "log10p_257":["mean","std"],
        "rho_h_257":["mean","std"],
        "rho_v_257":["mean","std"],
        "rho_d_257":["mean","std"],
        "rho_a_257":["mean","std"],
        })
        df_sum .columns =["StageKey"]+[f"{a}_{b}"for a ,b in df_sum .columns .tolist ()[1 :]]
        df_sum ["Stage"]=df_sum ["StageKey"]
        df_sum =df_sum [["Stage",*[c for c in df_sum .columns if c not in ("StageKey","Stage")]]]
        self .save_results (df_sum ,"stage_metrics_pf257_stage_summary.xlsx")

        # Paper Table* LaTeX
        tex =self ._build_paper_table_tex (df_long ,df_sanity )
        out_tex =self .output_dir /"stage_metrics_pf257_stage_summary_subtables.tex"
        out_tex .write_text (tex ,encoding ="utf-8")

        return df_long 
