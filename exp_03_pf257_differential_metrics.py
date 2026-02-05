"""
Experiment 03 - Differential security metrics.

Computes NPCR, UACI, AC and key-sensitivity proxies under controlled input
differences and key/nonce variations.
"""
from __future__ import annotations 

from typing import Any ,Dict ,List ,Tuple 
import sys 
from pathlib import Path 
import hashlib 
import zlib 
import secrets 

import numpy as np 
import pandas as pd 

from pf257_experiment_base import BaseExperiment # type: ignore


# ----------------------------------------------------------------------
# Popcount LUT (SciPy olmadan)
# ----------------------------------------------------------------------
_POP8 =np .array ([bin (i ).count ("1")for i in range (256 )],dtype =np .uint8 )
_POP9 =np .array ([bin (i ).count ("1")for i in range (512 )],dtype =np .uint8 )# 9-bit (0..511)


# ----------------------------------------------------------------------
# Metrikler
# ----------------------------------------------------------------------
def npcr (a :np .ndarray ,b :np .ndarray )->float :
    """NPCR (%)"""
    a =np .asarray (a )
    b =np .asarray (b )
    if a .shape !=b .shape :
        raise ValueError ("NPCR: shape mismatch.")
    return float (100.0 *np .mean (a !=b ))


def uaci_u8 (a_u8 :np .ndarray ,b_u8 :np .ndarray )->float :
    """UACI (%) for uint8 images (0..255)."""
    a =np .asarray (a_u8 ,dtype =np .int32 )
    b =np .asarray (b_u8 ,dtype =np .int32 )
    if a .shape !=b .shape :
        raise ValueError ("UACI(u8): shape mismatch.")
    return float (100.0 *np .mean (np .abs (a -b ))/255.0 )


def uaci_field257 (a_u16 :np .ndarray ,b_u16 :np .ndarray )->float :
    """
    UACI (%) for Z_257 images (values 0..256).
    Normalize by 256 (max absolute difference).
    """
    a =np .asarray (a_u16 ,dtype =np .int32 )
    b =np .asarray (b_u16 ,dtype =np .int32 )
    if a .shape !=b .shape :
        raise ValueError ("UACI(field257): shape mismatch.")
    return float (100.0 *np .mean (np .abs (a -b ))/256.0 )


def ac_bits_u8 (a_u8 :np .ndarray ,b_u8 :np .ndarray )->float :
    """Avalanche Criterion (%) for uint8 images (8 bit/pixel)."""
    a =np .asarray (a_u8 ,dtype =np .uint8 )
    b =np .asarray (b_u8 ,dtype =np .uint8 )
    if a .shape !=b .shape :
        raise ValueError ("AC(u8): shape mismatch.")
    xor =np .bitwise_xor (a ,b ).ravel ()
    changed =int (_POP8 [xor ].sum ())
    total =xor .size *8 
    return float (100.0 *changed /total )if total >0 else 0.0 


def ac_bits_field9 (a_u16 :np .ndarray ,b_u16 :np .ndarray )->float :
    """Avalanche Criterion (%) for Z_257 values using 9-bit representation (0..256)."""
    a =np .asarray (a_u16 ,dtype =np .uint16 )
    b =np .asarray (b_u16 ,dtype =np .uint16 )
    if a .shape !=b .shape :
        raise ValueError ("AC(field9): shape mismatch.")
    xor =(np .bitwise_xor (a ,b )&0x01FF ).astype (np .uint16 ).ravel ()
    changed =int (_POP9 [xor ].sum ())
    total =xor .size *9 
    return float (100.0 *changed /total )if total >0 else 0.0 


def field_to_u8_view (field2d :np .ndarray )->np .ndarray :
    """See README.md for usage and details."""
    f =np .asarray (field2d ,dtype =np .uint16 )
    return np .where (f ==256 ,255 ,f ).astype (np .uint8 ,copy =False )


def to_gray_u8 (img :np .ndarray )->np .ndarray :
    """See README.md for usage and details."""
    x =np .asarray (img )
    if x .ndim ==2 :
        return np .asarray (x ,dtype =np .uint8 ,order ="C")
    if x .ndim ==3 and x .shape [-1 ]in (3 ,4 ):
        g =np .mean (x [...,:3 ],axis =2 )
        return np .asarray (np .clip (np .rint (g ),0 ,255 ),dtype =np .uint8 ,order ="C")
    raise ValueError (f"Unsupported image shape: {x.shape}")


def flip_one_pixel (img_u8 :np .ndarray ,rng :np .random .Generator )->np .ndarray :
    """See README.md for usage and details."""
    out =np .array (img_u8 ,copy =True )
    H ,W =out .shape 
    i =int (rng .integers (0 ,H ))
    j =int (rng .integers (0 ,W ))
    old =int (out [i ,j ])
    new =int (rng .integers (0 ,256 ))
    if new ==old :
        new =(new +1 )&0xFF 
    out [i ,j ]=np .uint8 (new )
    return out 


    # ----------------------------------------------------------------------
    # Experiment 03
    # ----------------------------------------------------------------------
class PF257DifferentialMetrics (BaseExperiment ):
    """Differential security metrics (NPCR/UACI/AC, key sensitivity) (Experiment 03)."""

    NUM_REPEATS_DEFAULT :int =5 

    def __init__ (self ,config ,cipher ,**params :Any )->None :
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="(PF257) Differential metrics — NPCR/UACI/AC + Key/Nonce Sensitivity",
        )

        self .num_repeats =int (params .get ("num_repeats",self .NUM_REPEATS_DEFAULT ))

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _get_core (self ):
        obj =self .cipher 

        # NOTE: (comment removed; repository is English-only)
        if hasattr (obj ,"_encrypt_pixels")and hasattr (obj ,"_field_stream"):
            core =obj 
            bs =int (getattr (core ,"bs",0 ))
            return core ,bs 

            # 2) wrapper -> adapter -> core
        adapter =getattr (obj ,"cipher",None )
        if adapter is None :
            raise AttributeError ("See README.md for usage and details.")
        core =getattr (adapter ,"core",None )
        if core is None :
            raise AttributeError ("See README.md for usage and details.")
        bs =int (getattr (core ,"bs",0 ))
        return core ,bs 

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _make_plain_params (self ,core ,seed :int )->bytes :
        rng =np .random .default_rng (int (seed )&0xFFFFFFFF )
        seed16 =rng .integers (0 ,256 ,16 ,dtype =np .uint8 ).tobytes ()
        Kf =int (rng .integers (0 ,256 ,dtype =np .uint16 ))
        Kr =int (rng .integers (0 ,256 ,dtype =np .uint16 ))
        key32 =rng .integers (0 ,256 ,32 ,dtype =np .uint8 ).tobytes ()
        gk =[int (x )for x in rng .integers (1 ,256 ,int (core .bs ),dtype =np .uint16 )]# 1..255
        return core ._pack_params (seed16 ,Kf ,Kr ,key32 ,gk )# type: ignore[attr-defined]

    @staticmethod 
    def _flip_one_bit (buf :bytes ,pos :int ,bit :int =0 )->bytes :
        if not (0 <=bit <=7 ):
            raise ValueError ("bit must be in [0..7].")
        if not (0 <=pos <len (buf )):
            raise ValueError ("pos out of range.")
        b =bytearray (buf )
        b [pos ]^=(1 <<bit )
        return bytes (b )

    def _params_seed16_bitflip (self ,plain_params :bytes )->bytes :
        return self ._flip_one_bit (plain_params ,pos =0 ,bit =0 )

    def _params_key32_bitflip (self ,plain_params :bytes )->bytes :
    # layout: seed16(16) + Kf(1) + Kr(1) + key32(32) + gk(bs)
        return self ._flip_one_bit (plain_params ,pos =18 ,bit =0 )

        # ------------------------------------------------------------------
        # Deterministik stage encryption
        # ------------------------------------------------------------------
    def _det_encrypt_stages (
    self ,
    core ,
    img_u8 :np .ndarray ,
    plain_params :bytes ,
    nonce :int ,
    )->Tuple [np .ndarray ,np .ndarray ,np .ndarray ]:
        S ,_INV ,Kf ,Kr ,key32 ,gk_vec =core ._unpack_params (plain_params )# type: ignore[attr-defined]

        # Pixel layer
        c2_u8 =core ._encrypt_pixels (img_u8 ,S ,Kf ,Kr )# type: ignore[attr-defined]
        H ,W =c2_u8 .shape 

        bs =int (core .bs )
        Hp =H +((-H )%bs )
        Wp =W +((-W )%bs )
        pixel_pad_u8 =core ._pad_edge_u8 (c2_u8 ,Hp ,Wp )# type: ignore[attr-defined]

        # Field convert
        c2_field =pixel_pad_u8 .astype (np .uint16 ,copy =False )

        # NOTE: (comment removed; repository is English-only)
        blk =core ._blocks_of (c2_field )# type: ignore[attr-defined]
        ni ,nj ,_ ,_ =blk .shape 
        blk_flat =blk .reshape ((ni *nj ,bs ,bs ))

        V ,U =core ._vu_from_gk (gk_vec )# type: ignore[attr-defined]
        mixed_flat =core ._mix_affine (blk_flat ,V ,U )# type: ignore[attr-defined]
        mixed_blk =mixed_flat .reshape ((ni ,nj ,bs ,bs ))
        mixed_field =core ._unblocks (mixed_blk ,Hp ,Wp )# type: ignore[attr-defined]

        # Stream mask (Z_257)
        stream =core ._field_stream (key32 ,Hp *Wp ,int (nonce )).reshape (Hp ,Wp ).astype (np .int32 )# type: ignore[attr-defined]
        masked =(mixed_field .astype (np .int32 )+stream )%int (core .P_FIELD )
        masked_field =masked .astype (np .uint16 )

        return pixel_pad_u8 ,mixed_field ,masked_field 

    @staticmethod 
    def _nonce_for_image (seed :int ,img_name :str )->int :
        h =hashlib .sha256 (f"PF257_NONCE|exp03|{int(seed)}|{img_name}".encode ("utf-8")).digest ()
        return int .from_bytes (h [:8 ],"big",signed =False )

    @staticmethod 
    def _flip_rng (seed :int ,img_name :str )->np .random .Generator :
        s =(int (seed )^(zlib .crc32 (img_name .encode ("utf-8"))&0xFFFFFFFF )^0x03D1FF )&0xFFFFFFFF 
        return np .random .default_rng (s )

        # ------------------------------------------------------------------
        # RUN
        # ------------------------------------------------------------------
    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Differential metrics (PF257) started...")

        core ,bs =self ._get_core ()
        base_seed =int (getattr (self .config ,"seed",0 ))
        deterministic =bool (getattr (self .config ,"deterministic",False ))
        R =int (max (1 ,self .num_repeats ))

        # NOTE: (comment removed; repository is English-only)
        params_seed =(base_seed ^0xA53C )&0xFFFFFFFF 
        plain_params =self ._make_plain_params (core ,params_seed )
        plain_params_seedflip =self ._params_seed16_bitflip (plain_params )
        plain_params_keyflip =self ._params_key32_bitflip (plain_params )

        rows :List [Dict [str ,Any ]]=[]

        for img_name ,img_orig in images .items ():
            img_u8 =to_gray_u8 (img_orig )
            H ,W =img_u8 .shape 

            if deterministic :
                nonce =self ._nonce_for_image (base_seed ,img_name )
            else :
                nonce =secrets .randbits (64 )

            self .log (f"Processing {img_name} ({R} repeats)...")

            # Referans stage'ler
            pix1 ,mix1 ,post1 =self ._det_encrypt_stages (core ,img_u8 ,plain_params ,nonce )

            # NOTE: (comment removed; repository is English-only)
            change_rate_prepost =float (np .mean (mix1 !=post1 ))

            # Key sensitivity (1-bit flip)
            _pix_seedflip ,_mix_seedflip ,post_seedflip =self ._det_encrypt_stages (core ,img_u8 ,plain_params_seedflip ,nonce )
            _pix_keyflip ,_mix_keyflip ,post_keyflip =self ._det_encrypt_stages (core ,img_u8 ,plain_params_keyflip ,nonce )

            key_seed_npcr =npcr (post1 ,post_seedflip )
            key_seed_uaci =uaci_field257 (post1 ,post_seedflip )
            key_seed_ac =ac_bits_field9 (post1 ,post_seedflip )

            key_stream_npcr =npcr (post1 ,post_keyflip )
            key_stream_uaci =uaci_field257 (post1 ,post_keyflip )
            key_stream_ac =ac_bits_field9 (post1 ,post_keyflip )

            # Nonce sensitivity
            _pix_n2 ,_mix_n2 ,post_nonce2 =self ._det_encrypt_stages (core ,img_u8 ,plain_params ,(nonce +1 )&0xFFFFFFFFFFFFFFFF )
            nonce_npcr =npcr (post1 ,post_nonce2 )
            nonce_uaci =uaci_field257 (post1 ,post_nonce2 )
            nonce_ac =ac_bits_field9 (post1 ,post_nonce2 )

            # Plaintext differential (tek-piksel flip)
            flip_rng =self ._flip_rng (base_seed ,img_name )

            pix_npcr ,pix_uaci ,pix_ac =[],[],[]
            mix_npcr ,mix_uaci ,mix_ac =[],[],[]
            post_npcr ,post_uaci ,post_ac =[],[],[]
            post_npcr_u8 ,post_uaci_u8 ,post_ac_u8 =[],[],[]

            post1_u8 =field_to_u8_view (post1 )

            for _ in range (R ):
                img_flip =flip_one_pixel (img_u8 ,flip_rng )
                pix2 ,mix2 ,post2 =self ._det_encrypt_stages (core ,img_flip ,plain_params ,nonce )

                # PIXEL
                pix_npcr .append (npcr (pix1 ,pix2 ))
                pix_uaci .append (uaci_u8 (pix1 ,pix2 ))
                pix_ac .append (ac_bits_u8 (pix1 ,pix2 ))

                # MIXED (Z_257)
                mix_npcr .append (npcr (mix1 ,mix2 ))
                mix_uaci .append (uaci_field257 (mix1 ,mix2 ))
                mix_ac .append (ac_bits_field9 (mix1 ,mix2 ))

                # MASKED (Z_257)
                post_npcr .append (npcr (post1 ,post2 ))
                post_uaci .append (uaci_field257 (post1 ,post2 ))
                post_ac .append (ac_bits_field9 (post1 ,post2 ))

                # MASKED uint8 view (ikincil)
                post2_u8 =field_to_u8_view (post2 )
                post_npcr_u8 .append (npcr (post1_u8 ,post2_u8 ))
                post_uaci_u8 .append (uaci_u8 (post1_u8 ,post2_u8 ))
                post_ac_u8 .append (ac_bits_u8 (post1_u8 ,post2_u8 ))

            rows .append ({
            "Image":img_name ,
            "Shape":f"{int(H)}x{int(W)}",
            "PadShape":f"{int(pix1.shape[0])}x{int(pix1.shape[1])}",
            "bs":int (bs ),
            "Nonce64":int (nonce ),

            "ChangeRate_PrePost":float (change_rate_prepost ),

            "NPCR_PIXEL_%":float (np .mean (pix_npcr )),
            "UACI_PIXEL_%":float (np .mean (pix_uaci )),
            "AC_PIXEL_%":float (np .mean (pix_ac )),

            "NPCR_MIXED_%":float (np .mean (mix_npcr )),
            "UACI_MIXED_%":float (np .mean (mix_uaci )),
            "AC_MIXED_%":float (np .mean (mix_ac )),

            "NPCR_MASKED_%":float (np .mean (post_npcr )),
            "UACI_MASKED_%":float (np .mean (post_uaci )),
            "AC_MASKED_%":float (np .mean (post_ac )),

            "NPCR_MASKED_u8_%":float (np .mean (post_npcr_u8 )),
            "UACI_MASKED_u8_%":float (np .mean (post_uaci_u8 )),
            "AC_MASKED_u8_%":float (np .mean (post_ac_u8 )),

            "NPCR_KEY_seed16_%":float (key_seed_npcr ),
            "UACI_KEY_seed16_%":float (key_seed_uaci ),
            "AC_KEY_seed16_%":float (key_seed_ac ),

            "NPCR_KEY_stream_%":float (key_stream_npcr ),
            "UACI_KEY_stream_%":float (key_stream_uaci ),
            "AC_KEY_stream_%":float (key_stream_ac ),

            "NPCR_NONCE_%":float (nonce_npcr ),
            "UACI_NONCE_%":float (nonce_uaci ),
            "AC_NONCE_%":float (nonce_ac ),
            })

            self .log (
            f"  {img_name}: NPCR_MASKED={np.mean(post_npcr):.4f}%, "
            f"UACI_MASKED={np.mean(post_uaci):.4f}%, "
            f"ChangeRate_PrePost={change_rate_prepost:.6f}"
            )

        cols =[
        "Image","Shape","PadShape","bs","Nonce64",
        "ChangeRate_PrePost",
        "NPCR_PIXEL_%","UACI_PIXEL_%","AC_PIXEL_%",
        "NPCR_MIXED_%","UACI_MIXED_%","AC_MIXED_%",
        "NPCR_MASKED_%","UACI_MASKED_%","AC_MASKED_%",
        "NPCR_MASKED_u8_%","UACI_MASKED_u8_%","AC_MASKED_u8_%",
        "NPCR_KEY_seed16_%","UACI_KEY_seed16_%","AC_KEY_seed16_%",
        "NPCR_KEY_stream_%","UACI_KEY_stream_%","AC_KEY_stream_%",
        "NPCR_NONCE_%","UACI_NONCE_%","AC_NONCE_%",
        ]
        df =pd .DataFrame (rows ,columns =cols )

        self .save_results (df ,"differential_metrics_pf257.xlsx")
        self .export_latex (
        df .copy (),
        caption ="Differential metrics for PF257-EGKEM-SPNCTR (avg of repeated one-pixel flips).",
        label ="tab:diff_pf257",
        )

        self .log ("See README.md for usage and details.")
        return df 
