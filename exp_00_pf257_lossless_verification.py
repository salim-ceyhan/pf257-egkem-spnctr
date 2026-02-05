"""
Experiment 00 - Lossless encryption/decryption verification.

Verifies pixel-perfect reconstruction (MSE=0, PSNR=∞) when decrypting a packet
produced by the PF257-EGKEM-SPNCTR cipher core.
"""
from __future__ import annotations 

from typing import Dict ,Any 
import sys 
from pathlib import Path 
import hashlib 
import numpy as np 
import pandas as pd 
import secrets 

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import LosslessMetrics ,_POPCOUNT 


class PF257LosslessVerification (BaseExperiment ):

    """Lossless encryption/decryption verification (Experiment 00)."""


    def __init__ (self ,config ,cipher ,**_params :Any ):
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 

        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")
        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="PF257 lossless encryption-decryption verification",
        )

    @staticmethod 
    def _bit_errors (a :np .ndarray ,b :np .ndarray )->int :
        xor =np .bitwise_xor (a .ravel (),b .ravel ())
        return int (_POPCOUNT [xor ].sum ())

    def _nonce64 (self ,img_name :str )->int :
        """See README.md for usage and details."""
        if bool (getattr (self .config ,"deterministic",False )):
            seed =int (getattr (self .config ,"seed",42 ))
            h =hashlib .sha256 (f"PF257_NONCE|{seed}|{img_name}".encode ("utf-8")).digest ()
            return int .from_bytes (h [:8 ],"big",signed =False )
        return int (secrets .randbits (64 ))

    @staticmethod 
    def _to_gray_u8 (img :np .ndarray )->np .ndarray :
        """See README.md for usage and details."""
        x =np .asarray (img )
        if x .ndim ==2 :
            return np .asarray (x ,dtype =np .uint8 ,order ="C")
        if x .ndim ==3 and x .shape [-1 ]in (3 ,4 ):
        # NOTE: (comment removed; repository is English-only)
            g =np .mean (x [...,:3 ],axis =2 )
            return np .asarray (np .clip (np .rint (g ),0 ,255 ),dtype =np .uint8 ,order ="C")
        raise ValueError (f"Unsupported image shape for lossless test: {x.shape}")

    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Losslessness verification started...")

        rows =[]
        for img_name ,img_orig in images .items ():
            self .log (f"Processing {img_name}...")

            img_u8 =self ._to_gray_u8 (img_orig )

            # NOTE: (comment removed; repository is English-only)
            nonce64 =self ._nonce64 (img_name )

            packet ,bob_priv ,_ =self .cipher .encrypt (
            img_u8 ,
            nonce =nonce64 ,
            return_priv =True ,
            )

            img_dec =self .cipher .decrypt (packet ,bob_priv =bob_priv ,verify =True )

            if img_u8 .shape !=img_dec .shape :
                self .log (f"  HATA: shape mismatch: {img_u8.shape} != {img_dec.shape}")
                img_dec =np .zeros_like (img_u8 )

            mse =LosslessMetrics .mse (img_u8 ,img_dec )
            psnr =LosslessMetrics .psnr (img_u8 ,img_dec )
            pixel_errors =int (np .sum (img_u8 !=img_dec ))
            bit_errors =self ._bit_errors (img_u8 ,img_dec )

            ok =(pixel_errors ==0 )and (bit_errors ==0 )and (mse <1e-12 )

            rows .append (
            {
            "Image":img_name ,
            "H":int (img_u8 .shape [0 ]),
            "W":int (img_u8 .shape [1 ]),
            "Nonce64":int (nonce64 ),
            "MSE":float (mse ),
            "PSNR_dB":float (psnr ),
            "Pixel_Errors":int (pixel_errors ),
            "Bit_Errors":int (bit_errors ),
            "Lossless":bool (ok ),
            }
            )

            ps ="∞"if np .isinf (psnr )else f"{psnr:.2f}"
            self .log (f"  {img_name}: MSE={mse:.6f}, PSNR={ps} dB [{'PASS' if ok else 'FAIL'}]")

        df =pd .DataFrame (rows )
        self .save_results (df ,"lossless_verification_pf257.xlsx")

        all_ok =bool (df ["Lossless"].all ())if len (df )else False 
        self .log (f"Summary: {'All PASS' if all_ok else 'FAILED'}")
        if not all_ok :
            self .log ("See README.md for usage and details.")

        return df 