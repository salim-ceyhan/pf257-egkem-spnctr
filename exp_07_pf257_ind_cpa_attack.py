# exp_07_pf257_ind_cpa_attack.py
"""See README.md for usage and details."""

from __future__ import annotations 

from typing import Any ,Dict ,List ,Tuple ,Optional ,Union 
import sys 
from pathlib import Path 
import secrets 
import struct 

import numpy as np 
import pandas as pd 

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import ExperimentConfig 


class PF257INDCpaHistogramExperiment (BaseExperiment ):
    """Experiment 07 (PF257): LR-IND-CPA^nr distinguishing — histogram L1 + AUC (final attacker-visible output)."""

    ROUNDS :int =5_000 
    TRAIN :int =512 
    CROP :Tuple [int ,int ]=(128 ,128 )
    PAIR_MODE :str ="crop_vs_invert"# {"crop_vs_invert", "two_crops"}

    def __init__ (self ,config :ExperimentConfig ,cipher ,**params :Any )->None :
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="PF257 LR-IND-CPA^nr (histogram L1) + AUC on attacker-visible final output",
        )

        rounds =params .get ("grid_rounds")
        if isinstance (rounds ,list )and rounds :
            self .ROUNDS =int (rounds [0 ])

        train =params .get ("grid_train")
        if isinstance (train ,list )and train :
            self .TRAIN =int (train [0 ])

        crop =params .get ("grid_crops")
        if isinstance (crop ,list )and crop :
            self .CROP =tuple (crop [0 ])# type: ignore[assignment]

        pm =params .get ("pair_mode")
        if isinstance (pm ,str )and pm .strip ():
            self .PAIR_MODE =pm .strip ()

            # ----------------------------- helpers -----------------------------

    @staticmethod 
    def _rng (seed :int )->np .random .Generator :
        return np .random .default_rng (int (seed )&0xFFFFFFFF )

    @staticmethod 
    def _wilson_upper (correct :int ,trials :int ,z :float )->float :
        """See README.md for usage and details."""
        if trials <=0 :
            return float ("nan")
        n =float (trials )
        phat =float (correct )/n 
        denom =1.0 +(z *z )/n 
        center =(phat +(z *z )/(2.0 *n ))/denom 
        rad =z *np .sqrt ((phat *(1.0 -phat )/n )+(z *z )/(4.0 *n *n ))/denom 
        return float (center +rad )

    @staticmethod 
    def _auc_mann_whitney (scores_pos :np .ndarray ,scores_neg :np .ndarray )->float :
        """See README.md for usage and details."""
        x =np .asarray (scores_pos ,dtype =np .float64 )
        y =np .asarray (scores_neg ,dtype =np .float64 )
        n1 =int (x .size )
        n0 =int (y .size )
        if n1 ==0 or n0 ==0 :
            return 0.5 

        z =np .concatenate ([x ,y ])
        order =np .argsort (z ,kind ="mergesort")
        z_sorted =z [order ]

        ranks_sorted =np .empty_like (z_sorted ,dtype =np .float64 )
        i =0 
        rank =1.0 
        N =z_sorted .size 
        while i <N :
            j =i +1 
            while j <N and z_sorted [j ]==z_sorted [i ]:
                j +=1 
            avg =(rank +(rank +(j -i )-1.0 ))/2.0 
            ranks_sorted [i :j ]=avg 
            rank +=(j -i )
            i =j 

        ranks =np .empty_like (ranks_sorted ,dtype =np .float64 )
        ranks [order ]=ranks_sorted 

        ranks_pos_sum =float (np .sum (ranks [:n1 ]))
        U =ranks_pos_sum -(n1 *(n1 +1 )/2.0 )
        auc =U /float (n1 *n0 )
        return float (max (0.0 ,min (1.0 ,auc )))

    @staticmethod 
    def _fresh_nonce64 (used :set [int ])->int :
        """See README.md for usage and details."""
        while True :
            n =int (secrets .randbits (64 ))
            if n not in used :
                used .add (n )
                return n 

                # ---------------------- attacker-visible view ----------------------

    @staticmethod 
    def _largest_payload_candidate (packet :dict )->Tuple [Optional [np .ndarray ],Optional [bytes ]]:
        """See README.md for usage and details."""
        best_arr :Optional [np .ndarray ]=None 
        best_arr_nbytes =-1 

        best_bytes :Optional [bytes ]=None 
        best_bytes_len =-1 

        for _k ,v in packet .items ():
            if isinstance (v ,(bytes ,bytearray )):
                bl =len (v )
                if bl >best_bytes_len :
                    best_bytes_len =bl 
                    best_bytes =bytes (v )
            else :
                try :
                    arr =np .asarray (v )
                except Exception :
                    continue 
                if arr .ndim >=1 and arr .dtype .kind in ("u","i"):
                    nb =int (arr .nbytes )
                    if nb >best_arr_nbytes :
                        best_arr_nbytes =nb 
                        best_arr =arr 

        return best_arr ,best_bytes 

    @staticmethod 
    def _try_extract_field257 (packet :dict )->Optional [np .ndarray ]:
        """See README.md for usage and details."""
        # NOTE: (comment removed; repository is English-only)
        for key in ("C_field","c_field","C","cipher_field","ct_field","C_blocks_field"):
            v =packet .get (key ,None )
            if v is None :
                continue 
            try :
                arr =np .asarray (v )
            except Exception :
                continue 
            if arr .dtype .kind in ("u","i"):
                a =arr .astype (np .int64 ,copy =False ).ravel ()
                if a .size >0 and int (a .min ())>=0 and int (a .max ())<=256 :
                    return a .astype (np .uint16 ,copy =False )

                    # NOTE: (comment removed; repository is English-only)
        best_arr ,_ =PF257INDCpaHistogramExperiment ._largest_payload_candidate (packet )
        if best_arr is None :
            return None 
        if best_arr .dtype .kind not in ("u","i"):
            return None 

        a =best_arr .astype (np .int64 ,copy =False ).ravel ()
        if a .size ==0 :
            return None 
        if int (a .min ())>=0 and int (a .max ())<=256 :
            return a .astype (np .uint16 ,copy =False )

        return None 

    @staticmethod 
    def _packet_to_bytes_deterministic (packet :dict )->bytes :
        """See README.md for usage and details."""
        chunks :List [bytes ]=[]

        for k in sorted (packet .keys (),key =lambda x :str (x )):
            v =packet [k ]
            chunks .append (str (k ).encode ("utf-8")+b"=")

            if isinstance (v ,(bytes ,bytearray )):
                b =bytes (v )
                chunks .append (struct .pack ("<I",len (b )))
                chunks .append (b )
            elif isinstance (v ,(int ,np .integer )):
                chunks .append (struct .pack ("<q",int (v )))
            else :
                try :
                    arr =np .asarray (v )
                    if arr .dtype .kind in ("u","i","f")and arr .ndim >=1 :
                        b =arr .tobytes (order ="C")
                        chunks .append (struct .pack ("<I",len (b )))
                        chunks .append (b )
                    else :
                        s =str (v ).encode ("utf-8",errors ="ignore")
                        chunks .append (struct .pack ("<I",len (s )))
                        chunks .append (s )
                except Exception :
                    s =str (v ).encode ("utf-8",errors ="ignore")
                    chunks .append (struct .pack ("<I",len (s )))
                    chunks .append (s )

            chunks .append (b";")
        return b"".join (chunks )

        # ------------------------- hist + distance --------------------------

    @staticmethod 
    def _hist_norm_field257 (x_field_flat_u16 :np .ndarray )->np .ndarray :
        v =np .asarray (x_field_flat_u16 ,dtype =np .uint16 ).ravel ()
        h =np .bincount (v .astype (np .int64 ,copy =False ),minlength =257 ).astype (np .float64 )
        s =float (max (1.0 ,h .sum ()))
        return h /s 

    @staticmethod 
    def _hist_norm_bytes256 (b :bytes )->np .ndarray :
        v =np .frombuffer (b ,dtype =np .uint8 )
        h =np .bincount (v .astype (np .int64 ,copy =False ),minlength =256 ).astype (np .float64 )
        s =float (max (1.0 ,h .sum ()))
        return h /s 

    @staticmethod 
    def _l1 (p :np .ndarray ,q :np .ndarray )->float :
        return float (np .sum (np .abs (p -q )))

        # ------------------------------ run -------------------------------

    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log (
        "LR-IND-CPA^nr (hist L1) + AUC started. "
        "[TARGET: FINAL attacker-visible output / no debug intermediate]"
        )

        rows :List [Dict [str ,Any ]]=[]

        seed =int (getattr (self .config ,"seed",0 ))
        rng =self ._rng (seed )

        crop_h ,crop_w =int (self .CROP [0 ]),int (self .CROP [1 ])

        for name ,img in images .items ():
            img_u8 =np .asarray (img ,dtype =np .uint8 ,order ="C")
            H ,W =img_u8 .shape 

            if H <crop_h or W <crop_w :
                self .log (f"SKIPPING {name}: Image {H}x{W} smaller than CROP {crop_h}x{crop_w}.")
                continue 

                # NOTE: (comment removed; repository is English-only)
            top0 =int (rng .integers (0 ,H -crop_h +1 ))
            left0 =int (rng .integers (0 ,W -crop_w +1 ))
            A =img_u8 [top0 :top0 +crop_h ,left0 :left0 +crop_w ].copy ()

            if self .PAIR_MODE =="two_crops":
                top1 =int (rng .integers (0 ,H -crop_h +1 ))
                left1 =int (rng .integers (0 ,W -crop_w +1 ))
                B =img_u8 [top1 :top1 +crop_h ,left1 :left1 +crop_w ].copy ()
            else :
            # NOTE: (comment removed; repository is English-only)
                B =(255 -A ).astype (np .uint8 ,copy =False )

            used_nonces :set [int ]=set ()

            # NOTE: (comment removed; repository is English-only)
            def enc_packet (P :np .ndarray ,nonce64 :int )->dict :
                enc_res =self .cipher .encrypt (
                np .asarray (P ,dtype =np .uint8 ,order ="C"),
                nonce =int (nonce64 ),
                return_priv =False ,
                debug_return_intermediate =False ,
                )
                if isinstance (enc_res ,tuple ):
                    packet =enc_res [0 ]
                else :
                    packet =enc_res 
                if not isinstance (packet ,dict ):
                    raise RuntimeError ("See README.md for usage and details.")
                return packet 

                # NOTE: (comment removed; repository is English-only)
            view_mode ="field257"
            h0_acc :Optional [np .ndarray ]=None 
            h1_acc :Optional [np .ndarray ]=None 

            # NOTE: (comment removed; repository is English-only)
            nonce_probe =self ._fresh_nonce64 (used_nonces )
            pkt_probe =enc_packet (A ,nonce_probe )
            field_probe =self ._try_extract_field257 (pkt_probe )

            if field_probe is None :
                view_mode ="bytes256"

            def packet_to_hist (packet :dict )->np .ndarray :
                if view_mode =="field257":
                    x_field =self ._try_extract_field257 (packet )
                    if x_field is None :
                    # NOTE: (comment removed; repository is English-only)
                        b =self ._packet_to_bytes_deterministic (packet )
                        return self ._hist_norm_bytes256 (b )
                    return self ._hist_norm_field257 (x_field )
                else :
                    best_arr ,best_bytes =self ._largest_payload_candidate (packet )
                    if best_bytes is not None :
                        return self ._hist_norm_bytes256 (best_bytes )
                    b =self ._packet_to_bytes_deterministic (packet )
                    return self ._hist_norm_bytes256 (b )

                    # NOTE: (comment removed; repository is English-only)
            for _ in range (int (self .TRAIN )):
                n0 =self ._fresh_nonce64 (used_nonces )
                n1 =self ._fresh_nonce64 (used_nonces )

                h0 =packet_to_hist (enc_packet (A ,n0 ))
                h1 =packet_to_hist (enc_packet (B ,n1 ))

                if h0_acc is None :
                    h0_acc =np .zeros_like (h0 ,dtype =np .float64 )
                if h1_acc is None :
                    h1_acc =np .zeros_like (h1 ,dtype =np .float64 )

                h0_acc +=h0 
                h1_acc +=h1 

            h0_tpl =h0_acc /float (max (1 ,self .TRAIN ))# type: ignore
            h1_tpl =h1_acc /float (max (1 ,self .TRAIN ))# type: ignore

            # ----------------- LR test (ROUNDS) -----------------
            correct =0 
            scores_pos :List [float ]=[]# bit==0 (P0=A)
            scores_neg :List [float ]=[]# bit==1 (P1=B)

            for _ in range (int (self .ROUNDS )):
                bit =int (rng .integers (0 ,2 ))
                P =A if bit ==0 else B 

                nonce64 =self ._fresh_nonce64 (used_nonces )
                hC =packet_to_hist (enc_packet (P ,nonce64 ))

                d0 =self ._l1 (hC ,h0_tpl )
                d1 =self ._l1 (hC ,h1_tpl )

                pred =0 if d0 <d1 else 1 
                correct +=int (pred ==bit )

                # NOTE: (comment removed; repository is English-only)
                score =(d1 -d0 )
                if bit ==0 :
                    scores_pos .append (score )
                else :
                    scores_neg .append (score )

            acc =float (correct )/float (self .ROUNDS )
            adv =float (abs (acc -0.5 ))
            auc =self ._auc_mann_whitney (np .asarray (scores_pos ),np .asarray (scores_neg ))

            wil95 =self ._wilson_upper (correct ,int (self .ROUNDS ),z =1.959963984540054 )
            wil99 =self ._wilson_upper (correct ,int (self .ROUNDS ),z =2.5758293035489004 )
            adv_ul95 =float (max (0.0 ,wil95 -0.5 ))
            adv_ul99 =float (max (0.0 ,wil99 -0.5 ))

            rows .append ({
            "Image":str (name ),
            "Shape":f"{int(H)}x{int(W)}",
            "Rounds":int (self .ROUNDS ),
            "Train":int (self .TRAIN ),
            "Crop":f"{int(crop_h)}x{int(crop_w)}",
            "PairMode":str (self .PAIR_MODE ),
            "ViewMode":str (view_mode ),
            "Acc":float (acc ),
            "Adv":float (adv ),
            "Acc_Wilson95_UL":float (wil95 ),
            "Acc_Wilson99_UL":float (wil99 ),
            "Adv_UL95":float (adv_ul95 ),
            "Adv_UL99":float (adv_ul99 ),
            "AUC":float (auc ),
            })

            self .log (
            f"  {name}: View={view_mode}, Acc={acc:.4f}, Adv={adv:.4f}, AUC={auc:.4f} "
            f"(train={self.TRAIN}, rounds={self.ROUNDS}, crop={crop_h}x{crop_w})"
            )

        df =pd .DataFrame (
        rows ,
        columns =[
        "Image","Shape","Rounds","Train","Crop","PairMode","ViewMode",
        "Acc","Adv","Acc_Wilson95_UL","Acc_Wilson99_UL","Adv_UL95","Adv_UL99","AUC",
        ],
        )

        self .save_results (df ,"ind_cpa_lr_nr_pf257.xlsx")
        return df 
