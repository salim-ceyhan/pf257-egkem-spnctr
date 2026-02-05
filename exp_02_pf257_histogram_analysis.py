# exp_02_pf257_histogram_analysis.py
from __future__ import annotations 

from typing import Dict ,Any ,Tuple ,List 
import sys 
from pathlib import Path 
import hashlib 
import secrets 
import math 

import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import CiphertextView 


def _to_gray_u8 (img :np .ndarray )->np .ndarray :
    x =np .asarray (img )
    if x .ndim ==2 :
        return np .asarray (x ,dtype =np .uint8 ,order ="C")
    if x .ndim ==3 and x .shape [-1 ]in (3 ,4 ):
        g =np .mean (x [...,:3 ],axis =2 )
        return np .asarray (np .clip (np .rint (g ),0 ,255 ),dtype =np .uint8 ,order ="C")
    raise ValueError (f"Unsupported image shape: {x.shape}")


def _entropy (x :np .ndarray ,alphabet :int )->float :
    v =np .asarray (x ).ravel ()
    if v .size ==0 :
        return 0.0 
    hist =np .bincount (v .astype (np .int64 ,copy =False ),minlength =int (alphabet )).astype (np .float64 ,copy =False )
    nz =hist [hist >0 ]
    p =nz /float (nz .sum ())
    return float (-(p *np .log2 (p )).sum ())


def _gammaincc (a :float ,x :float ,*,itmax :int =200 ,eps :float =3e-14 )->float :
    if a <=0 :
        raise ValueError ("gammaincc requires a>0")
    if x <=0 :
        return 1.0 

    gln =math .lgamma (a )
    if x <a +1.0 :
        ap =a 
        summ =1.0 /a 
        delt =summ 
        for _ in range (1 ,itmax +1 ):
            ap +=1.0 
            delt *=x /ap 
            summ +=delt 
            if abs (delt )<abs (summ )*eps :
                break 
        p =summ *math .exp (-x +a *math .log (x )-gln )
        q =1.0 -p 
        return float (min (1.0 ,max (0.0 ,q )))

    fpm =1e-300 
    b =x +1.0 -a 
    c =1.0 /fpm 
    d =1.0 /max (b ,fpm )
    h =d 
    for i in range (1 ,itmax +1 ):
        an =-float (i )*(float (i )-a )
        b +=2.0 
        d =an *d +b 
        if abs (d )<fpm :
            d =fpm 
        c =b +an /c 
        if abs (c )<fpm :
            c =fpm 
        d =1.0 /d 
        delt =d *c 
        h *=delt 
        if abs (delt -1.0 )<eps :
            break 

    q =math .exp (-x +a *math .log (x )-gln )*h 
    return float (min (1.0 ,max (0.0 ,q )))


def _chi2_uniform (x :np .ndarray ,alphabet :int )->Tuple [float ,float ,int ]:
    v =np .asarray (x ).ravel ()
    n =int (v .size )
    if n ==0 :
        return 0.0 ,1.0 ,int (alphabet -1 )

    a =int (alphabet )
    obs =np .bincount (v .astype (np .int64 ,copy =False ),minlength =a ).astype (np .float64 ,copy =False )
    exp =n /float (a )
    chi2 =float (((obs -exp )**2 /exp ).sum ())
    df =int (a -1 )
    p =float (_gammaincc (df /2.0 ,chi2 /2.0 ))
    return chi2 ,p ,df 


def _field_to_u8_view (field2d :np .ndarray )->np .ndarray :
    f =np .asarray (field2d ,dtype =np .uint16 ,order ="C")
    return np .where (f ==256 ,255 ,f ).astype (np .uint8 ,copy =False )


class PF257HistogramAnalysis (BaseExperiment ):
    def __init__ (self ,config ,cipher ,**_params :Any ):
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="Histogram/uniformity analysis (Z_257 view)",
        )

    def _nonce64 (self ,img_name :str )->int :
    # NOTE: (comment removed; repository is English-only)
        if bool (getattr (self .config ,"deterministic",True )):
            seed =int (getattr (self .config ,"seed",0 ))
            h =hashlib .blake2b (f"PF257_NONCE|EXP02|{seed}|{img_name}".encode ("utf-8"),digest_size =8 ).digest ()
            nonce =int .from_bytes (h ,"big",signed =False )&((1 <<63 )-1 )
            return 1 if nonce ==0 else int (nonce )
        nonce =int (secrets .randbits (63 ))
        return 1 if nonce ==0 else nonce 

    @staticmethod 
    def _crop (img :np .ndarray ,shape :Tuple [int ,int ])->np .ndarray :
        H ,W =int (shape [0 ]),int (shape [1 ])
        return img [:H ,:W ]

    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Histogram analysis (PF257) started...")

        rows :List [Dict [str ,Any ]]=[]

        for img_name ,img_orig in images .items ():
            self .log (f"Processing {img_name}...")

            P =_to_gray_u8 (img_orig )
            H ,W =P .shape 

            nonce =self ._nonce64 (img_name )

            # NOTE: (comment removed; repository is English-only)
            packet ,out =self .cipher .encrypt (P ,nonce =int (nonce ))

            blocks4d =packet .get ("blocks",None )
            if blocks4d is None :
                if isinstance (out ,np .ndarray )and out .ndim ==4 :
                    blocks4d =out 
                else :
                    raise ValueError ("Ciphertext blocks missing (packet['blocks']).")

                    # NOTE: (comment removed; repository is English-only)
            C_field_pad =CiphertextView .blocks4d_to_field2d (np .asarray (blocks4d ,dtype =np .uint16 ))

            # NOTE: (comment removed; repository is English-only)
            C_field =self ._crop (C_field_pad ,(H ,W ))
            C_view =_field_to_u8_view (C_field )

            ent_P_256 =_entropy (P ,256 )
            chi2_P_256 ,p_P_256 ,df256 =_chi2_uniform (P ,256 )

            ent_C_257 =_entropy (C_field ,257 )
            chi2_C_257 ,p_C_257 ,df257 =_chi2_uniform (C_field ,257 )

            ent_Cview_256 =_entropy (C_view ,256 )
            chi2_Cview_256 ,p_Cview_256 ,_ =_chi2_uniform (C_view ,256 )

            rows .append ({
            "Image":img_name ,
            "H":int (H ),"W":int (W ),
            "Nonce64":int (nonce ),

            "Entropy_256_P":float (ent_P_256 ),
            "Chi2_256_P":float (chi2_P_256 ),
            "Chi2_df_256_P":int (df256 ),
            "Chi2_p_256_P":float (p_P_256 ),

            "Entropy_257_Cfield":float (ent_C_257 ),
            "Chi2_257_Cfield":float (chi2_C_257 ),
            "Chi2_df_257_Cfield":int (df257 ),
            "Chi2_p_257_Cfield":float (p_C_257 ),

            "Entropy_256_Cview":float (ent_Cview_256 ),
            "Chi2_256_Cview":float (chi2_Cview_256 ),
            "Chi2_p_256_Cview":float (p_Cview_256 ),
            })

            # NOTE: (comment removed; repository is English-only)
            try :
                fig ,axes =plt .subplots (3 ,1 ,figsize =(10 ,9 ))

                histP =np .bincount (P .ravel ().astype (np .int64 ,copy =False ),minlength =256 )
                axes [0 ].plot (histP )
                axes [0 ].set_title ("P — 256 bins (crop)")
                axes [0 ].set_xlabel ("0..255");axes [0 ].set_ylabel ("Count")

                histC =np .bincount (C_field .ravel ().astype (np .int64 ,copy =False ),minlength =257 )
                axes [1 ].plot (histC )
                axes [1 ].set_title ("C_field — 257 bins (Z_257, crop)")
                axes [1 ].set_xlabel ("0..256");axes [1 ].set_ylabel ("Count")

                histCv =np .bincount (C_view .ravel ().astype (np .int64 ,copy =False ),minlength =256 )
                axes [2 ].plot (histCv )
                axes [2 ].set_title ("C_view — 256 bins (projeksiyon, crop)")
                axes [2 ].set_xlabel ("0..255");axes [2 ].set_ylabel ("Count")

                plt .tight_layout ()
                out_png =self .output_dir /f"{Path(img_name).stem}_hist_pf257.png"
                plt .savefig (out_png ,dpi =150 )
                plt .close (fig )
            except Exception as e :
                self .log (f"See README.md for usage and details.")

        df =pd .DataFrame (rows )
        self .save_results (df ,"histogram_analysis_pf257.xlsx")

        self .log ("See README.md for usage and details.")
        return df 
