"""
pf257_experiment_framework.py
Modular experiment framework for the PF257-EGKEM-SPNCTR suite.

Components:
- ExperimentConfig: global configuration (output directory, block size, etc.)
- ImageIO: loading grayscale test images from a directory
- StatisticalMetrics / helpers: entropy, chi-square, correlations, etc.
- CipherWrapper: stable adapter around the cipher core for the experiments
"""
from __future__ import annotations 

from dataclasses import dataclass 
from math import erfc ,log10 
from pathlib import Path 
from typing import Any ,Dict ,List ,Optional ,Tuple ,Union 

import numpy as np 
import pandas as pd 
from PIL import Image 
from inspect import signature ,Parameter 


# =========================
# CONFIG
# =========================
@dataclass 
class ExperimentConfig :
    block_size :int =16 
    deterministic :bool =True 
    seed :int =42 
    use_gpu :bool =False 
    output_dir :Path =Path ("RESULTS_PF257_PAPER")

    # NOTE: (comment removed; repository is English-only)
    analysis_alphabet :int =256 

    def __post_init__ (self )->None :
        self .output_dir =Path (self .output_dir )
        self .output_dir .mkdir (parents =True ,exist_ok =True )
        if self .analysis_alphabet not in (256 ,257 ):
            raise ValueError ("analysis_alphabet must be 256 or 257.")


            # =========================
            # IMAGE I/O
            # =========================
class ImageIO :
    @staticmethod 
    def load_grayscale (path :Union [str ,Path ])->np .ndarray :
        img =Image .open (path ).convert ("L")
        return np .asarray (img ,dtype =np .uint8 ,order ="C")

    @staticmethod 
    def load_test_images (image_dir :Union [str ,Path ])->Dict [str ,np .ndarray ]:
        image_dir =Path (image_dir )
        images :Dict [str ,np .ndarray ]={}
        exts ={".tiff",".tif",".png",".jpg",".jpeg",".bmp"}
        if not image_dir .exists ():
            print (f"⚠ Directory not found: {image_dir}")
            return images 
        files =[p for p in sorted (image_dir .iterdir ())if p .is_file ()and p .suffix .lower ()in exts ]
        if not files :
            print (f"⚠ No images found in: {image_dir}")
            return images 
        for p in files :
            try :
                images [p .name ]=ImageIO .load_grayscale (p )
                print (f"✓ Loaded: {p.name}")
            except Exception as e :
                print (f"⚠ Failed to load {p.name}: {e}")
        return images 


        # =========================
        # CIPHERTEXT VIEW HELPERS
        # =========================
class CiphertextView :
    """See README.md for usage and details."""

    @staticmethod 
    def blocks4d_to_field2d (blocks :np .ndarray )->np .ndarray :
        if not (isinstance (blocks ,np .ndarray )and blocks .ndim ==4 ):
            raise ValueError ("Expected 4D blocks array.")
        ni ,nj ,bs ,_ =blocks .shape 
        Hp ,Wp =ni *bs ,nj *bs 
        # (ni,nj,bs,bs) -> (Hp,Wp)
        return blocks .swapaxes (1 ,2 ).reshape (Hp ,Wp ).astype (np .uint16 ,copy =False )

    @staticmethod 
    def field2d_to_u8 (field_img :np .ndarray )->np .ndarray :
        field =np .asarray (field_img ,dtype =np .uint16 )
        # NOTE: (comment removed; repository is English-only)
        u8 =np .where (field ==256 ,255 ,field ).astype (np .uint8 ,copy =False )
        return u8 

    @staticmethod 
    def coerce_output_to_u8 (out :Any ,packet :Any )->Any :
        """See README.md for usage and details."""
        if isinstance (out ,np .ndarray ):
            if out .ndim ==2 and out .dtype ==np .uint8 :
                return out 
            if out .ndim ==2 and out .dtype in (np .uint16 ,np .int32 ,np .int64 ,np .uint32 ):
                return CiphertextView .field2d_to_u8 (out )
            if out .ndim ==4 and out .dtype in (np .uint16 ,np .int32 ,np .int64 ,np .uint32 ):
                field2d =CiphertextView .blocks4d_to_field2d (out .astype (np .uint16 ,copy =False ))
                return CiphertextView .field2d_to_u8 (field2d )
        return out 


        # =========================
        # CIPHER WRAPPER (PF257 uyumlu)
        # =========================
class CipherWrapper :
    """See README.md for usage and details."""
    def __init__ (self ,cipher_core :Any )->None :
        self .cipher :Any =cipher_core 

    def encrypt (self ,img :np .ndarray ,**kwargs :Any )->Any :
        fn =getattr (self .cipher ,"encrypt")
        sig =signature (fn )
        names =list (sig .parameters .keys ())

        # NOTE: (comment removed; repository is English-only)
        if "bob_pub_h"in names and "bob_pub_h"not in kwargs :
            for attr in ("bob_pub_h","receiver_pub_h","pub_h"):
                if hasattr (self .cipher ,attr ):
                    kwargs ["bob_pub_h"]=getattr (self .cipher ,attr )
                    break 

        res =fn (np .asarray (img ,dtype =np .uint8 ,order ="C"),**kwargs )

        # NOTE: (comment removed; repository is English-only)
        if isinstance (res ,tuple )and len (res )==2 :
            packet ,out =res 
            out2 =CiphertextView .coerce_output_to_u8 (out ,packet )
            return packet ,out2 

        if isinstance (res ,tuple )and len (res )==3 :
            packet ,priv ,out =res 
            out2 =CiphertextView .coerce_output_to_u8 (out ,packet )
            return packet ,priv ,out2 

        return res 

    def decrypt (self ,packet :Any ,bob_priv :Optional [bytes ]=None ,*,verify :bool =True )->np .ndarray :
        fn =getattr (self .cipher ,"decrypt")
        sig =signature (fn )
        names =[p .name for p in sig .parameters .values ()]
        kwargs :Dict [str ,Any ]={}

        if "verify"in names :
            kwargs ["verify"]=verify 

        if bob_priv is not None :
        # NOTE: (comment removed; repository is English-only)
            for alt in (
            "bob_priv_x","receiver_priv_x","priv_x","sk_int",
            "bob_priv","receiver_priv","priv","sk","private_key","secret_key"
            ):
                if alt in names :
                    kwargs [alt ]=bob_priv 
                    break 

        try :
        # NOTE: (comment removed; repository is English-only)
            if len (names )==1 :
                return fn (packet )

                # NOTE: (comment removed; repository is English-only)
            if len (names )>=2 and bob_priv is not None :
                try :
                    return fn (packet ,bob_priv ,**{k :v for k ,v in kwargs .items ()if k !="verify"})
                except TypeError :
                    return fn (packet ,**kwargs )

                    # 3) Sadece keyword ile dene
            return fn (packet ,**kwargs )

        except TypeError :
        # NOTE: (comment removed; repository is English-only)
            if bob_priv is not None :
                return fn (packet ,bob_priv )
            return fn (packet )

            # NOTE: (comment removed; repository is English-only)
    def _derive_params_compat (self ,shared :bytes ,salt :bytes ,ctx_tag :bytes )->Any :
        fn =getattr (self .cipher ,"_derive_params",None )
        if fn is None :
            raise AttributeError ("Cipher core has no _derive_params")
        sig =signature (fn )
        params =tuple (sig .parameters .values ())
        n =len (params )
        if n ==2 :
            return fn (shared ,salt )
        if n ==3 :
            p_ctx =params [2 ]
            if p_ctx .default is not Parameter .empty :
                return fn (shared ,salt )
            return fn (shared ,salt ,ctx_tag )
        raise TypeError (f"Unsupported _derive_params signature with {n} parameters")


        # =========================
        # STAT METRICS (256/257 parametrik)
        # =========================
class StatisticalMetrics :
    @staticmethod 
    def entropy (img :np .ndarray ,*,alphabet :int =256 )->float :
        x =np .asarray (img ).ravel ()
        hist ,_ =np .histogram (x ,bins =alphabet ,range =(0 ,alphabet ))
        hist =hist [hist >0 ]
        p =hist /hist .sum ()if hist .size else np .array ([1.0 ])
        return float (-np .sum (p *np .log2 (p )))

    @staticmethod 
    def chi_square (img :np .ndarray ,*,alphabet :int =256 )->Tuple [float ,float ]:
        x =np .asarray (img ).ravel ()
        hist ,_ =np .histogram (x ,bins =alphabet ,range =(0 ,alphabet ))
        N =float (x .size )
        expected =N /float (alphabet )
        chi2 =float (np .sum ((hist -expected )**2 /(expected +1e-12 )))

        k =float (alphabet -1 )
        z =((chi2 /k )**(1.0 /3.0 )-(1.0 -2.0 /(9.0 *k )))/np .sqrt (2.0 /(9.0 *k ))
        z =float (np .clip (z ,-10.0 ,10.0 ))
        p_value =0.5 *erfc (z /np .sqrt (2.0 ))
        return chi2 ,float (p_value )

    @staticmethod 
    def neighbor_correlations (img :np .ndarray )->Dict [str ,float ]:
        img =np .asarray (img )
        H ,W =img .shape 

        def corr (a ,b ):
            a =a .astype (np .float64 ).ravel ()
            b =b .astype (np .float64 ).ravel ()
            ma ,mb =a .mean (),b .mean ()
            sa ,sb =a .std (),b .std ()
            if sa <1e-12 or sb <1e-12 :
                return 0.0 
            return float (np .mean ((a -ma )*(b -mb ))/(sa *sb ))

        rho_h =corr (img [:,:-1 ],img [:,1 :])if W >1 else 0.0 
        rho_v =corr (img [:-1 ,:],img [1 :,:])if H >1 else 0.0 
        rho_d =corr (img [:-1 ,:-1 ],img [1 :,1 :])if (H >1 and W >1 )else 0.0 
        rho_a =corr (img [:-1 ,1 :],img [1 :,:-1 ])if (H >1 and W >1 )else 0.0 
        return {"rho_h":rho_h ,"rho_v":rho_v ,"rho_d":rho_d ,"rho_a":rho_a }


        # =========================
        # NOTE: (comment removed; repository is English-only)
        # =========================
_POPCOUNT =np .unpackbits (
np .arange (256 ,dtype =np .uint8 )[:,None ],axis =1 
).sum (axis =1 ).astype (np .uint8 )

class DifferentialMetrics :
    @staticmethod 
    def npcr_uaci (C1 :np .ndarray ,C2 :np .ndarray ,*,max_val :float =255.0 )->Tuple [float ,float ]:
        if C1 .shape !=C2 .shape :
            raise ValueError ("Images must have same shape")
        N =C1 .size 
        npcr =(float (np .sum (C1 !=C2 ))/N )*100.0 
        uaci =(float (np .sum (np .abs (C1 .astype (np .float64 )-C2 .astype (np .float64 ))))/(N *max_val ))*100.0 
        return npcr ,uaci 

    @staticmethod 
    def avalanche_criterion (C1 :np .ndarray ,C2 :np .ndarray )->float :
        if C1 .shape !=C2 .shape :
            raise ValueError ("Images must have same shape")
        xor =np .bitwise_xor (C1 .astype (np .uint8 ).ravel (),C2 .astype (np .uint8 ).ravel ())
        changed_bits =int (_POPCOUNT [xor ].sum ())
        total_bits =C1 .size *8 
        return (changed_bits /total_bits )*100.0 

    @staticmethod 
    def flip_one_pixel (img :np .ndarray )->np .ndarray :
        H ,W =img .shape 
        out =np .array (img ,copy =True ,dtype =np .uint8 ,order ="C")
        i ,j =H //2 ,W //2 
        out [i ,j ]^=1 
        return out 


        # =========================
        # LOSSLESS METRICS
        # =========================
class LosslessMetrics :
    @staticmethod 
    def mse (img1 :np .ndarray ,img2 :np .ndarray )->float :
        if img1 .shape !=img2 .shape :
            raise ValueError ("Images must have same shape")
        d =img1 .astype (np .float64 )-img2 .astype (np .float64 )
        return float (np .mean (d *d ))

    @staticmethod 
    def psnr (img1 :np .ndarray ,img2 :np .ndarray ,max_val :float =255.0 )->float :
        m =LosslessMetrics .mse (img1 ,img2 )
        if m <1e-10 :
            return float ("inf")
        return 10.0 *log10 ((max_val *max_val )/m )


        # =========================
        # LATEX EXPORT
        # =========================
class LaTeXExporter :
    @staticmethod 
    def dataframe_to_latex (df :pd .DataFrame ,caption :str ,label :str ,float_format :str =".4f")->str :
        lines =[]
        lines +=[r"\begin{table}[htbp]\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}"]
        col_spec ="l"+"c"*(len (df .columns )-1 )
        lines +=[rf"\begin{{tabular}}{{{col_spec}}}",r"\toprule"]
        lines +=[" & ".join ([rf"\textbf{{{c}}}"for c in df .columns ])+r" \\",r"\midrule"]
        for _ ,row in df .iterrows ():
            cells :List [str ]=[]
            for v in row :
                if isinstance (v ,(int ,np .integer )):
                    cells .append (str (int (v )))
                elif isinstance (v ,(float ,np .floating )):
                    if np .isinf (v ):
                        cells .append (r"$\infty$")
                    elif np .isnan (v ):
                        cells .append ("---")
                    else :
                        cells .append (f"{float(v):{float_format}}")
                else :
                    cells .append (str (v ))
            lines +=[" & ".join (cells )+r" \\"]
        lines +=[r"\bottomrule",r"\end{tabular}",r"\end{table}"]
        return "\n".join (lines )

    @staticmethod 
    def save_latex_table (latex_code :str ,filepath :Union [str ,Path ])->None :
        fp =Path (filepath )
        fp .parent .mkdir (parents =True ,exist_ok =True )
        fp .write_text (latex_code ,encoding ="utf-8")
        print (f"✓ LaTeX table saved: {fp}")


        # =========================
        # BASIC VALIDATION
        # =========================
def validate_experiment_setup (config :ExperimentConfig ,images :Dict [str ,np .ndarray ])->None :
    print ("\n"+"="*70 )
    print ("EXPERIMENT VALIDATION")
    print ("="*70 )
    assert config .block_size >0 ,"Block size must be positive"
    assert config .output_dir .exists (),"Output directory must exist"
    print (f"✓ Config: bs={config.block_size}, det={config.deterministic}, seed={config.seed}, alphabet={config.analysis_alphabet}")

    assert len (images )>0 ,"No images loaded"
    for name ,img in images .items ():
        assert img .ndim ==2 ,f"{name}: must be 2D grayscale"
        assert img .dtype ==np .uint8 ,f"{name}: must be uint8"
        H ,W =img .shape 
        pad_h =(-H )%config .block_size 
        pad_w =(-W )%config .block_size 
        if pad_h or pad_w :
            print (f"  ⚠ {name}: will be padded ({H}x{W} → {H+pad_h}x{W+pad_w})")

    print (f"✓ Images: {len(images)} loaded")
    for name ,img in images .items ():
        print (f"  - {name}: {img.shape}")
    print ("="*70 +"\n")
