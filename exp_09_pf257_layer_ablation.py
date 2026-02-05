# exp_09_pf257_layer_ablation.py
"""See README.md for usage and details."""

from __future__ import annotations 

from typing import Any ,Dict ,List ,Tuple ,Optional ,Callable ,cast 
import sys 
from pathlib import Path 
import math 
import io 
import gzip 
import secrets 

import numpy as np 
import pandas as pd 

import matplotlib .pyplot as plt 

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import StatisticalMetrics 

from cryptography .hazmat .primitives .asymmetric import x25519 
from cryptography .hazmat .primitives .serialization import Encoding ,PublicFormat 

try :
    from PIL import Image # type: ignore
    _HAS_PIL =True 
except Exception :
    _HAS_PIL =False 


class PF257LayerAblationStudy (BaseExperiment ):
    """Layer ablation study across pipeline variants (Experiment 09)."""

    DEFAULT_N_REPEATS :int =1 
    DEFAULT_MAKE_SCATTER :bool =True 
    DEFAULT_SCATTER_MAX_POINTS :int =5000 
    DEFAULT_ENABLE_COMPRESSIBILITY :bool =True # NOTE: (comment removed; repository is English-only)

    def __init__ (self ,config ,cipher ,**params :Any )->None :
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="PF257 layer ablation / contribution (pixel, mix, mask; u8-view + field stats)",
        )

        self .N_REPEATS =int (params .get ("n_repeats",self .DEFAULT_N_REPEATS ))
        self .MAKE_SCATTER =bool (params .get ("make_scatter",self .DEFAULT_MAKE_SCATTER ))
        self .SCATTER_MAX_POINTS =int (params .get ("scatter_max_points",self .DEFAULT_SCATTER_MAX_POINTS ))
        self .ENABLE_COMPRESSIBILITY =bool (params .get ("enable_compressibility",self .DEFAULT_ENABLE_COMPRESSIBILITY ))

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _get_pf257_core_ctx (self ):
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
    def _chi2_pvalue_wilson_hilferty (chi2 :float ,df :int )->float :
        """See README.md for usage and details."""
        if df <=0 or chi2 <0 :
            return float ("nan")
        d =float (df )
        t =(chi2 /d )**(1.0 /3.0 )
        mu =1.0 -2.0 /(9.0 *d )
        sigma =math .sqrt (2.0 /(9.0 *d ))
        z =(t -mu )/max (sigma ,1e-15 )
        # p = 1 - Phi(z)
        p =1.0 -0.5 *(1.0 +math .erf (z /math .sqrt (2.0 )))
        return float (max (0.0 ,min (1.0 ,p )))

    @classmethod 
    def _chi2_uniform (cls ,x :np .ndarray ,alphabet :int )->Tuple [float ,int ,float ]:
        """See README.md for usage and details."""
        v =np .asarray (x ).ravel ()
        n =int (v .size )
        if alphabet <=1 :
            return float ("nan"),0 ,float ("nan")
        if n ==0 :
            return 0.0 ,int (alphabet -1 ),1.0 

        obs =np .bincount (v .astype (np .int64 ,copy =False ),minlength =alphabet ).astype (np .float64 )
        exp =n /float (alphabet )
        chi2 =float (np .sum ((obs -exp )**2 /(exp +1e-12 )))
        df =int (alphabet -1 )
        p =cls ._chi2_pvalue_wilson_hilferty (chi2 ,df =df )
        return chi2 ,df ,p 

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _field_to_u8_view (field2d :np .ndarray )->np .ndarray :
        """See README.md for usage and details."""
        f =np .asarray (field2d ,dtype =np .uint16 ,order ="C")
        out =f .copy ()
        out [out ==256 ]=255 
        return out .astype (np .uint8 ,copy =False )

        # ------------------------------------------------------------------
        # Pad (core ile compatible olacak figurede)
        # ------------------------------------------------------------------
    @staticmethod 
    def _pad_edge_u8 (core ,img_u8 :np .ndarray ,Hp :int ,Wp :int )->np .ndarray :
        if hasattr (core ,"_pad_edge_u8")and callable (getattr (core ,"_pad_edge_u8")):
            return np .asarray (core ._pad_edge_u8 (img_u8 ,int (Hp ),int (Wp )),dtype =np .uint8 ,order ="C")
        H ,W =img_u8 .shape 
        if H ==Hp and W ==Wp :
            return np .asarray (img_u8 ,dtype =np .uint8 ,order ="C")
        return np .pad (
        np .asarray (img_u8 ,dtype =np .uint8 ,order ="C"),
        ((0 ,Hp -H ),(0 ,Wp -W )),
        mode ="edge",
        )

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _compressibility_ratios_u8 (arr_u8 :np .ndarray )->Tuple [float ,float ]:
        raw =np .asarray (arr_u8 ,dtype =np .uint8 ,order ="C").tobytes ()
        raw_size =max (1 ,len (raw ))

        png_ratio =float ("nan")
        if _HAS_PIL :
            try :
                buf =io .BytesIO ()
                Image .fromarray (np .asarray (arr_u8 ,dtype =np .uint8 )).save (buf ,format ="PNG",optimize =True )# type: ignore
                png_ratio =len (buf .getvalue ())/raw_size 
            except Exception :
                png_ratio =float ("nan")

        gzip_ratio =float ("nan")
        try :
            gz =gzip .compress (raw ,compresslevel =9 )
            gzip_ratio =len (gz )/raw_size 
        except Exception :
            gzip_ratio =float ("nan")

        return float (png_ratio ),float (gzip_ratio )

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _save_scatter (self ,arr_u8 :np .ndarray ,title :str ,seed :int )->None :
        arr_u8 =np .asarray (arr_u8 ,dtype =np .uint8 ,order ="C")
        H ,W =arr_u8 .shape 
        if W <2 :
            return 

        x =arr_u8 [:,:-1 ].ravel ()
        y =arr_u8 [:,1 :].ravel ()
        n =int (x .size )
        k =int (min (self .SCATTER_MAX_POINTS ,n ))

        rng =np .random .default_rng (int (seed )&0xFFFFFFFF )
        if k <n :
            idx =rng .choice (n ,size =k ,replace =False )
            xs ,ys =x [idx ],y [idx ]
        else :
            xs ,ys =x ,y 

        plt .figure (figsize =(4 ,4 ),dpi =110 )
        plt .scatter (xs ,ys ,s =1 ,alpha =0.25 ,edgecolors ="none",rasterized =True )
        plt .title (title ,fontsize =9 )
        plt .xlabel ("x (left pixel)",fontsize =8 )
        plt .ylabel ("y (right pixel)",fontsize =8 )
        plt .xticks (fontsize =8 )
        plt .yticks (fontsize =8 )
        plt .xlim (-5 ,260 )
        plt .ylim (-5 ,260 )
        plt .grid (False )

        out =self .output_dir /f"{title.replace(' ', '_').replace('/', '_')}.png"
        plt .tight_layout (pad =0.4 )
        plt .savefig (out ,dpi =140 ,bbox_inches ="tight")
        plt .close ()
        self .log (f"  ✓ Scatter saved: {out.name}")

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _packet_get (packet :Dict [str ,Any ],*keys :str ):
        for k in keys :
            if k in packet and packet [k ]is not None :
                return packet [k ]
        return None 

    def _unwrap_and_unpack_params (
    self ,
    *,
    core ,
    packet :Dict [str ,Any ],
    bob_priv :bytes ,
    ):
        """See README.md for usage and details."""
        kem_epk =self ._packet_get (packet ,"kem_epk","epk")
        salt =self ._packet_get (packet ,"salt")
        enc_params =self ._packet_get (packet ,"enc_params","params_enc","params")

        if kem_epk is None or salt is None or enc_params is None :
            raise KeyError ("See README.md for usage and details.")

        unwrap =getattr (core ,"_unwrap_params",None )
        unpack =getattr (core ,"_unpack_params",None )
        ctx_fn =getattr (core ,"_ctx",None )
        if not (callable (unwrap )and callable (unpack )and callable (ctx_fn )):
            raise RuntimeError ("See README.md for usage and details.")

            # NOTE: (comment removed; repository is English-only)
        bob_pub_raw =getattr (self .cipher ,"bob_pub_raw",None )
        if not isinstance (bob_pub_raw ,(bytes ,bytearray )):
            bob_pub_raw =(
            x25519 .X25519PrivateKey .from_private_bytes (bytes (bob_priv ))
            .public_key ()
            .public_bytes (Encoding .Raw ,PublicFormat .Raw )
            )

        shared =(
        x25519 .X25519PrivateKey .from_private_bytes (bytes (bob_priv ))
        .exchange (x25519 .X25519PublicKey .from_public_bytes (bytes (kem_epk )))
        )
        ctx =ctx_fn (bytes (bob_pub_raw ),bytes (kem_epk ))

        params_plain =unwrap (bytes (shared ),bytes (salt ),ctx ,bytes (enc_params ))
        S ,_INV ,Kf ,Kr ,key32 ,gk_vec =unpack (params_plain )# type: ignore
        return S ,int (Kf ),int (Kr ),bytes (key32 ),np .asarray (gk_vec ,dtype =np .uint16 ,order ="C")

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _compute_layers (self ,img_u8 :np .ndarray ,*,nonce64 :int )->Dict [str ,np .ndarray ]:
        """See README.md for usage and details."""
        core ,_adapter =self ._get_pf257_core_ctx ()

        img_u8 =np .asarray (img_u8 ,dtype =np .uint8 ,order ="C")
        if img_u8 .ndim !=2 :
            raise ValueError ("Bu experiment 2D (grayscale) img_u8 bekler.")

            # NOTE: (comment removed; repository is English-only)
        enc_res =self .cipher .encrypt (
        img_u8 ,
        nonce =int (nonce64 ),
        return_priv =True ,
        debug_return_intermediate =False ,
        )
        if not isinstance (enc_res ,tuple ):
            raise RuntimeError ("See README.md for usage and details.")

        if len (enc_res )==3 :
            packet ,bob_priv ,_out =enc_res 
        elif len (enc_res )==2 :
            packet ,bob_priv =enc_res 
        else :
            raise RuntimeError (f"See README.md for usage and details.")

        if not isinstance (packet ,dict ):
            raise RuntimeError ("See README.md for usage and details.")
        if not isinstance (bob_priv ,(bytes ,bytearray )):
            raise RuntimeError ("See README.md for usage and details.")

        orig_shape =self ._packet_get (packet ,"orig_shape")or img_u8 .shape 
        pad_shape =self ._packet_get (packet ,"pad_shape")or img_u8 .shape 

        H ,W =int (orig_shape [0 ]),int (orig_shape [1 ])
        Hp ,Wp =int (pad_shape [0 ]),int (pad_shape [1 ])

        # NOTE: (comment removed; repository is English-only)
        S ,Kf ,Kr ,key32 ,gk_vec =self ._unwrap_and_unpack_params (core =core ,packet =packet ,bob_priv =bytes (bob_priv ))

        # Pad plaintext (core ile uyumlu)
        P_pad =self ._pad_edge_u8 (core ,img_u8 ,Hp ,Wp )

        # (S) Pixel layer
        enc_pix =getattr (core ,"_encrypt_pixels",None )
        if not callable (enc_pix ):
            raise RuntimeError ("See README.md for usage and details.")

        try :
            S_pad =enc_pix (P_pad ,S ,int (Kf ),int (Kr ))
        except TypeError :
            S_pad =enc_pix (P_pad ,int (Kf ),int (Kr ),S )
        S_pad_u8 =np .asarray (S_pad ,dtype =np .uint8 ,order ="C")

        # u8 -> field (0..255) -> uint16
        if hasattr (core ,"_u8_to_field")and callable (getattr (core ,"_u8_to_field")):
            u8_to_field =cast (Callable [[np .ndarray ],np .ndarray ],getattr (core ,"_u8_to_field"))
            P_field =np .asarray (u8_to_field (P_pad ),dtype =np .uint16 ,order ="C")
            S_field =np .asarray (u8_to_field (S_pad_u8 ),dtype =np .uint16 ,order ="C")
        else :
            P_field =P_pad .astype (np .uint16 ,copy =False )
            S_field =S_pad_u8 .astype (np .uint16 ,copy =False )

            # (SP) Mix
        blocks_of =getattr (core ,"_blocks_of",None )
        unblocks =getattr (core ,"_unblocks",None )
        vu_from_gk =getattr (core ,"_vu_from_gk",None )
        mix_affine =getattr (core ,"_mix_affine",None )
        if not (callable (blocks_of )and callable (unblocks )and callable (vu_from_gk )and callable (mix_affine )):
            raise RuntimeError ("See README.md for usage and details.")

        blk4d =blocks_of (S_field )# (ni,nj,bs,bs)
        ni ,nj ,bs ,_ =blk4d .shape # type: ignore
        blk3d =blk4d .reshape (int (ni *nj ),int (bs ),int (bs )).astype (np .uint16 ,copy =False )# type: ignore

        V ,U =vu_from_gk (gk_vec )# type: ignore
        mixed3d =mix_affine (blk3d ,V ,U )
        mixed4d =np .asarray (mixed3d ,dtype =np .uint16 ).reshape (int (ni ),int (nj ),int (bs ),int (bs ))

        try :
            SP_field_pad =unblocks (mixed4d ,int (Hp ),int (Wp ))
        except TypeError :
            SP_field_pad =unblocks (mixed4d ,(int (Hp ),int (Wp )))
        SP_field_pad =np .asarray (SP_field_pad ,dtype =np .uint16 ,order ="C")

        # Stream (Z_257)
        field_stream =getattr (core ,"_field_stream",None )
        if not callable (field_stream ):
            raise RuntimeError ("See README.md for usage and details.")

        stream =np .asarray (field_stream (key32 ,int (Hp *Wp ),int (nonce64 )),dtype =np .uint16 ).reshape (Hp ,Wp )

        # (M-only) plaintext + stream mod 257
        M_only_field_pad =(P_field +stream )%257 

        # (SPM) full: mixed + stream mod 257
        SPM_field_pad =(SP_field_pad +stream )%257 

        # u8 views
        M_only_u8 =self ._field_to_u8_view (M_only_field_pad )
        SP_u8 =self ._field_to_u8_view (SP_field_pad )
        SPM_u8 =self ._field_to_u8_view (SPM_field_pad )

        return {
        "orig_shape":np .array ([H ,W ],dtype =np .int64 ),
        "pad_shape":np .array ([Hp ,Wp ],dtype =np .int64 ),

        "P_u8":P_pad ,
        "S_u8":S_pad_u8 ,

        "M_only_field":np .asarray (M_only_field_pad ,dtype =np .uint16 ,order ="C"),
        "SP_field":SP_field_pad ,
        "SPM_field":np .asarray (SPM_field_pad ,dtype =np .uint16 ,order ="C"),

        "M_only_u8":np .asarray (M_only_u8 ,dtype =np .uint8 ,order ="C"),
        "SP_u8":np .asarray (SP_u8 ,dtype =np .uint8 ,order ="C"),
        "SPM_u8":np .asarray (SPM_u8 ,dtype =np .uint8 ,order ="C"),
        }

        # ------------------------------------------------------------------
        # Metrikler
        # ------------------------------------------------------------------
    def _metrics_u8 (self ,x_u8 :np .ndarray )->Dict [str ,float ]:
        x_u8 =np .asarray (x_u8 ,dtype =np .uint8 ,order ="C")
        Hx =float (StatisticalMetrics .entropy (x_u8 ))
        corr =StatisticalMetrics .neighbor_correlations (x_u8 )
        chi2 ,p =StatisticalMetrics .chi_square (x_u8 )
        return {
        "Entropy_256":float (Hx ),
        "Chi2_256":float (chi2 ),
        "Chi2_df_256":255.0 ,
        "Chi2_p_256":float (p ),
        "rho_h":float (corr .get ("rho_h",float ("nan"))),
        "rho_v":float (corr .get ("rho_v",float ("nan"))),
        }

    def _metrics_field257 (self ,x_field :np .ndarray )->Dict [str ,float ]:
        x =np .asarray (x_field ,dtype =np .uint16 ,order ="C")
        hist =np .bincount (x .ravel ().astype (np .int64 ,copy =False ),minlength =257 ).astype (np .float64 )
        pr =hist /max (1.0 ,float (hist .sum ()))
        pr =pr [pr >0 ]
        Hx =float (-(pr *np .log2 (pr )).sum ())

        chi2 ,df ,pv =self ._chi2_uniform (x ,257 )
        return {
        "Entropy_257":float (Hx ),
        "Chi2_257":float (chi2 ),
        "Chi2_df_257":float (df ),
        "Chi2_p_257":float (pv ),
        }

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Layer ablation (PF257) started...")

        seed =int (getattr (self .config ,"seed",0 ))
        rng =np .random .default_rng (seed )

        rows :List [Dict [str ,Any ]]=[]

        for img_name ,img in images .items ():
            img_u8 =np .asarray (img ,dtype =np .uint8 ,order ="C")
            if img_u8 .ndim !=2 :
                raise ValueError (f"See README.md for usage and details.")
            H0 ,W0 =map (int ,img_u8 .shape )

            for r in range (max (1 ,int (self .N_REPEATS ))):
            # NOTE: (comment removed; repository is English-only)
                nonce64 =int (rng .integers (0 ,2 **64 ,dtype =np .uint64 ))

                layers =self ._compute_layers (img_u8 ,nonce64 =nonce64 )

                # crop to original
                P =np .asarray (layers ["P_u8"][:H0 ,:W0 ],dtype =np .uint8 ,order ="C")
                S =np .asarray (layers ["S_u8"][:H0 ,:W0 ],dtype =np .uint8 ,order ="C")

                M_only_field =np .asarray (layers ["M_only_field"][:H0 ,:W0 ],dtype =np .uint16 ,order ="C")
                SP_field =np .asarray (layers ["SP_field"][:H0 ,:W0 ],dtype =np .uint16 ,order ="C")
                SPM_field =np .asarray (layers ["SPM_field"][:H0 ,:W0 ],dtype =np .uint16 ,order ="C")

                M_only_u8 =np .asarray (layers ["M_only_u8"][:H0 ,:W0 ],dtype =np .uint8 ,order ="C")
                SP_u8 =np .asarray (layers ["SP_u8"][:H0 ,:W0 ],dtype =np .uint8 ,order ="C")
                SPM_u8 =np .asarray (layers ["SPM_u8"][:H0 ,:W0 ],dtype =np .uint8 ,order ="C")

                # u8-metrics
                mP =self ._metrics_u8 (P )
                mS =self ._metrics_u8 (S )
                mM =self ._metrics_u8 (M_only_u8 )
                mSP =self ._metrics_u8 (SP_u8 )
                mSPM =self ._metrics_u8 (SPM_u8 )

                # field-metrics (257)
                fM =self ._metrics_field257 (M_only_field )
                fSP =self ._metrics_field257 (SP_field )
                fSPM =self ._metrics_field257 (SPM_field )

                # compressibility (opsiyonel)
                if self .ENABLE_COMPRESSIBILITY :
                    pngP ,gzP =self ._compressibility_ratios_u8 (P )
                    pngS ,gzS =self ._compressibility_ratios_u8 (S )
                    pngM ,gzM =self ._compressibility_ratios_u8 (M_only_u8 )
                    pngSP ,gzSP =self ._compressibility_ratios_u8 (SP_u8 )
                    pngSPM ,gzSPM =self ._compressibility_ratios_u8 (SPM_u8 )
                else :
                    pngP =gzP =pngS =gzS =pngM =gzM =pngSP =gzSP =pngSPM =gzSPM =float ("nan")

                row :Dict [str ,Any ]={
                "Image":str (img_name ),
                "Repeat":int (r ),
                "H":int (H0 ),
                "W":int (W0 ),
                "Nonce64":int (nonce64 ),
                }

                def add (prefix :str ,d :Dict [str ,float ])->None :
                    for k ,v in d .items ():
                        row [f"{prefix}_{k}"]=float (v )

                add ("P",mP )
                add ("S",mS )
                add ("Monly_u8",mM )
                add ("SP_u8",mSP )
                add ("SPM_u8",mSPM )

                add ("Monly_f257",fM )
                add ("SP_f257",fSP )
                add ("SPM_f257",fSPM )

                row .update ({
                "P_PNG_ratio":float (pngP ),"P_GZIP_ratio":float (gzP ),
                "S_PNG_ratio":float (pngS ),"S_GZIP_ratio":float (gzS ),
                "Monly_PNG_ratio":float (pngM ),"Monly_GZIP_ratio":float (gzM ),
                "SP_PNG_ratio":float (pngSP ),"SP_GZIP_ratio":float (gzSP ),
                "SPM_PNG_ratio":float (pngSPM ),"SPM_GZIP_ratio":float (gzSPM ),
                })

                rows .append (row )

                self .log (
                f"{img_name} [r={r}] : "
                f"H(P)={row['P_Entropy_256']:.4f}, "
                f"H(SPM_u8)={row['SPM_u8_Entropy_256']:.4f}, "
                f"rho_h(SPM_u8)={row['SPM_u8_rho_h']:.4f}, rho_v(SPM_u8)={row['SPM_u8_rho_v']:.4f}"
                )

                # NOTE: (comment removed; repository is English-only)
                if self .MAKE_SCATTER and r ==0 :
                    stem =Path (str (img_name )).stem 
                    self ._save_scatter (S ,title =f"scatter_S_pixel_{stem}",seed =seed ^0x51A9 )
                    self ._save_scatter (SPM_u8 ,title =f"scatter_SPM_full_u8_{stem}",seed =seed ^0xA915 )

        df =pd .DataFrame (rows )
        self .save_results (df ,"layer_ablation_pf257.xlsx")
        return df 
