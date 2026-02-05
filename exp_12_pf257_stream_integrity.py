# exp_12_pf257_stream_integrity.py
# ---------------------------------------------------------------------
# Experiment 12 — Video/Stream Integrity (PF257 suite uyumlu, robust I/O)
# ---------------------------------------------------------------------
from __future__ import annotations 

from typing import Any ,Dict ,List ,Tuple ,Set ,Optional 
import sys 
from pathlib import Path 
import zlib 

import numpy as np 
import pandas as pd 

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import LosslessMetrics ,_POPCOUNT 


class PF257StreamIntegritySuite (BaseExperiment ):
    """Stream/video integrity checks across sequential frames (Experiment 12)."""

    def __init__ (
    self ,
    config ,
    cipher ,
    *,
    frames_per_seq :int =32 ,
    repeats :int =1 ,
    perturb_pct :float =0.01 ,
    replay_trials :int =64 ,
    frames :Optional [int ]=None ,# eski param compatibility
    chain_from_prev :bool =False ,# NOTE: (comment removed; repository is English-only)
    )->None :
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="See README.md for usage and details.",
        )

        self .frames_per_seq :int =int (frames if frames is not None else frames_per_seq )
        self .repeats :int =int (repeats )
        self .perturb_pct :float =float (perturb_pct )
        self .replay_trials :int =int (replay_trials )
        self .chain_from_prev :bool =bool (chain_from_prev )

        if self .frames_per_seq <1 :
            raise ValueError ("See README.md for usage and details.")
        if not (0.0 <self .perturb_pct <=1.0 ):
            raise ValueError ("See README.md for usage and details.")
        if self .repeats <1 :
            raise ValueError ("See README.md for usage and details.")
        if self .replay_trials <0 :
            raise ValueError ("See README.md for usage and details.")

            # ----------------------------- helpers -----------------------------

    @staticmethod 
    def _rng (seed :int )->np .random .Generator :
        return np .random .default_rng (int (seed )&0xFFFFFFFF )

    @staticmethod 
    def _stable_u32 (name :str )->int :
    # Python hash randomization yerine deterministik CRC32
        return int (zlib .crc32 (name .encode ("utf-8"))&0xFFFFFFFF )

    @staticmethod 
    def _bit_hd_percent_u8 (a :np .ndarray ,b :np .ndarray )->float :
        """See README.md for usage and details."""
        xa =np .asarray (a ,dtype =np .uint8 ,order ="C").ravel ()
        xb =np .asarray (b ,dtype =np .uint8 ,order ="C").ravel ()
        if xa .size ==0 or xb .size ==0 :
            return 0.0 
        n =int (min (xa .size ,xb .size ))
        xr =np .bitwise_xor (xa [:n ],xb [:n ])
        ones =int (_POPCOUNT [xr ].sum ())
        total =n *8 
        return 100.0 *float (ones )/float (total )

    @staticmethod 
    def _perturb_frame (img_u8 :np .ndarray ,rng :np .random .Generator ,pct :float )->np .ndarray :
        """See README.md for usage and details."""
        img_u8 =np .asarray (img_u8 ,dtype =np .uint8 ,order ="C")
        H ,W =img_u8 .shape 
        n =int (max (1 ,round (pct *H *W )))
        rr =rng .integers (0 ,H ,size =n ,dtype =np .int64 )
        cc =rng .integers (0 ,W ,size =n ,dtype =np .int64 )
        delta =rng .integers (1 ,255 ,size =n ,dtype =np .uint8 )
        out =img_u8 .copy ()
        out [rr ,cc ]=np .bitwise_xor (out [rr ,cc ],delta )
        return out 

        # NOTE: (comment removed; repository is English-only)

    def _encrypt_packet (
    self ,
    plain_u8 :np .ndarray ,
    nonce :int ,
    *,
    want_priv :bool =True ,
    )->Tuple [dict ,Optional [Any ],np .ndarray ]:
        """See README.md for usage and details."""
        img =np .asarray (plain_u8 ,dtype =np .uint8 ,order ="C")

        # 1) return_priv=True denenir
        if want_priv :
            try :
                res =self .cipher .encrypt (img ,nonce =int (nonce ),return_priv =True )
                if isinstance (res ,tuple ):
                    if len (res )==3 :
                        pkt ,bob_priv ,out_u8 =res 
                        return dict (pkt ),bob_priv ,np .asarray (out_u8 ,dtype =np .uint8 ,order ="C")
                    if len (res )==2 :
                        pkt ,bob_priv =res 
                        # NOTE: (comment removed; repository is English-only)
                        # Bu durumda tekrar normal encrypt denenir.
                        raise TypeError ("See README.md for usage and details.")
            except TypeError :
                pass 
            except Exception :
            # NOTE: (comment removed; repository is English-only)
                pass 

                # 2) normal encrypt
        res2 =self .cipher .encrypt (img ,nonce =int (nonce ))
        if not isinstance (res2 ,tuple )or len (res2 )<2 :
            raise RuntimeError ("See README.md for usage and details.")
        pkt2 ,out_u8_2 =res2 [0 ],res2 [1 ]
        return dict (pkt2 ),None ,np .asarray (out_u8_2 ,dtype =np .uint8 ,order ="C")

    def _decrypt_packet (
    self ,
    packet :dict ,
    bob_priv :Optional [Any ],
    *,
    verify :bool =True ,
    )->np .ndarray :
        """See README.md for usage and details."""
        if bob_priv is not None :
            try :
                out =self .cipher .decrypt (packet ,bob_priv ,verify =bool (verify ))
                return np .asarray (out ,dtype =np .uint8 ,order ="C")
            except TypeError :
            # NOTE: (comment removed; repository is English-only)
                pass 

        out2 =self .cipher .decrypt (packet ,verify =bool (verify ))
        return np .asarray (out2 ,dtype =np .uint8 ,order ="C")

        # ------------------------- one sequence runner -------------------------

    def _run_one_sequence (
    self ,
    base :np .ndarray ,
    rng :np .random .Generator ,
    )->Tuple [List [float ],List [int ],Dict [str ,Any ]]:
        """See README.md for usage and details."""
        base_u8 =np .asarray (base ,dtype =np .uint8 ,order ="C")

        # --- frame 0 ---
        nonce0 =int (rng .integers (0 ,2 **63 -1 ,dtype =np .int64 ))
        pkt0 ,priv0 ,c0 =self ._encrypt_packet (base_u8 ,nonce0 ,want_priv =True )
        dec0 =self ._decrypt_packet (pkt0 ,priv0 ,verify =True )
        if float (LosslessMetrics .mse (base_u8 ,dec0 ))>=1e-12 :
            raise AssertionError ("See README.md for usage and details.")

        nonces :List [int ]=[int (pkt0 .get ("nonce",nonce0 ))]
        prev_cipher =c0 
        hd_list :List [float ]=[]

        prev_plain =base_u8 

        for _ in range (max (0 ,self .frames_per_seq -1 )):
            src_plain =prev_plain if self .chain_from_prev else base_u8 
            nxt_plain =self ._perturb_frame (src_plain ,rng ,self .perturb_pct )

            nonce_next =int (rng .integers (0 ,2 **63 -1 ,dtype =np .int64 ))
            pkt ,priv ,ciph =self ._encrypt_packet (nxt_plain ,nonce_next ,want_priv =True )

            dec =self ._decrypt_packet (pkt ,priv ,verify =True )
            if float (LosslessMetrics .mse (nxt_plain ,dec ))>=1e-12 :
                raise AssertionError ("See README.md for usage and details.")

            hd_list .append (self ._bit_hd_percent_u8 (ciph ,prev_cipher ))
            prev_cipher =ciph 
            prev_plain =nxt_plain 

            nonces .append (int (pkt .get ("nonce",nonce_next )))

        replay_ctx ={"pkt0":pkt0 ,"priv0":priv0 }
        return hd_list ,nonces ,replay_ctx 

        # ------------------------------- run ------------------------------------

    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Experiment 12 (Stream integrity) started...")

        rows :List [Dict [str ,Any ]]=[]
        seed =int (getattr (self .config ,"seed",0 ))

        for name ,img0 in images .items ():
            self .log (f"Processing {name}...")

            base =np .asarray (img0 ,dtype =np .uint8 ,order ="C")
            if base .ndim !=2 :
                raise ValueError ("See README.md for usage and details.")

            H ,W =base .shape 
            shape_str =f"{int(H)}x{int(W)}"

            # NOTE: (comment removed; repository is English-only)
            img_rng =self ._rng (seed ^self ._stable_u32 (name ))

            all_hd :List [float ]=[]
            all_nonces :List [int ]=[]
            first_replay_ctx :Optional [Dict [str ,Any ]]=None 

            for r in range (self .repeats ):
                hd_list ,nonce_list ,replay_ctx =self ._run_one_sequence (base ,img_rng )
                all_hd .extend (hd_list )
                all_nonces .extend (nonce_list )
                if first_replay_ctx is None :
                    first_replay_ctx =replay_ctx 

                    # NOTE: (comment removed; repository is English-only)
            expected_nonces =int (self .repeats )*int (self .frames_per_seq )
            unique_nonces =int (len (set (all_nonces )))
            nonce_collisions =int (max (0 ,expected_nonces -unique_nonces ))

            # NOTE: (comment removed; repository is English-only)
            if all_hd :
                hd_mean =float (np .mean (all_hd ))
                hd_std =float (np .std (all_hd ))
                hd_min =float (np .min (all_hd ))
                hd_max =float (np .max (all_hd ))
                hd_n =int (len (all_hd ))
            else :
                hd_mean =float ("nan")
                hd_std =float ("nan")
                hd_min =float ("nan")
                hd_max =float ("nan")
                hd_n =0 

                # ---- Replay test ----
            replay_crypt_accepts =0 
            replay_crypt_exceptions =0 
            replay_app_accepts =0 
            replay_app_blocked =False 

            if first_replay_ctx is not None and self .replay_trials >0 :
                pkt0 =first_replay_ctx ["pkt0"]
                priv0 =first_replay_ctx .get ("priv0",None )
                rep_nonce =int (pkt0 .get ("nonce",-1 ))

                # NOTE: (comment removed; repository is English-only)
                nonce_cache :Set [int ]=set (all_nonces )

                for _ in range (self .replay_trials ):
                # NOTE: (comment removed; repository is English-only)
                    try :
                        _ =self ._decrypt_packet (pkt0 ,priv0 ,verify =True )
                        replay_crypt_accepts +=1 
                    except Exception :
                        replay_crypt_exceptions +=1 

                        # NOTE: (comment removed; repository is English-only)
                    if rep_nonce in nonce_cache :
                    # blocked
                        pass 
                    else :
                        replay_app_accepts +=1 
                        nonce_cache .add (rep_nonce )

                replay_app_blocked =(replay_app_accepts ==0 )

            rows .append ({
            "Image":name ,
            "Shape":shape_str ,
            "Frames_per_seq":int (self .frames_per_seq ),
            "Repeats":int (self .repeats ),
            "Perturb_pct":float (self .perturb_pct ),
            "Chain_from_prev":bool (self .chain_from_prev ),

            "HD_samples_n":int (hd_n ),
            "Cipher_HD_mean_%":float (hd_mean ),
            "Cipher_HD_std_%":float (hd_std ),
            "Cipher_HD_min_%":float (hd_min ),
            "Cipher_HD_max_%":float (hd_max ),

            "Nonce_expected":int (expected_nonces ),
            "Nonce_unique_count":int (unique_nonces ),
            "Nonce_collisions":int (nonce_collisions ),

            "Replay_trials":int (self .replay_trials ),
            "Replay_crypt_accepts":int (replay_crypt_accepts ),
            "Replay_crypt_exceptions":int (replay_crypt_exceptions ),
            "Replay_app_accepts":int (replay_app_accepts ),
            "Replay_app_blocked":bool (replay_app_blocked ),
            })

            self .log (
            f"  {name}: HD_mean={hd_mean:.3f}% (n={hd_n}) | "
            f"nonces(unique/exp)={unique_nonces}/{expected_nonces} (coll={nonce_collisions}) | "
            f"replay_crypt_accepts={replay_crypt_accepts}, replay_app_blocked={replay_app_blocked}"
            )

        df =pd .DataFrame (rows ,columns =[
        "Image","Shape",
        "Frames_per_seq","Repeats","Perturb_pct","Chain_from_prev",
        "HD_samples_n","Cipher_HD_mean_%","Cipher_HD_std_%","Cipher_HD_min_%","Cipher_HD_max_%",
        "Nonce_expected","Nonce_unique_count","Nonce_collisions",
        "Replay_trials","Replay_crypt_accepts","Replay_crypt_exceptions",
        "Replay_app_accepts","Replay_app_blocked",
        ])

        self .save_results (df ,"stream_integrity.xlsx")

        # NOTE: (comment removed; repository is English-only)
        try :
            self .export_latex (
            df [[
            "Image","Shape",
            "Frames_per_seq","Repeats","Perturb_pct",
            "Cipher_HD_mean_%","Cipher_HD_std_%",
            "Nonce_unique_count","Nonce_collisions",
            "Replay_app_blocked"
            ]],
            caption =(
            "See README.md for usage and details."
            "ve replay engelleme (uygulama seviyesi nonce-cache)."
            ),
            label ="tab:stream-integrity",
            )
        except Exception :
            pass 

        return df 


        # NOTE: (comment removed; repository is English-only)
        # 12: ExperimentMeta(
        #     cls=PF257StreamIntegritySuite,
        # NOTE: (comment removed; repository is English-only)
        #     params={"frames_per_seq": 32, "repeats": 3, "perturb_pct": 0.01, "replay_trials": 64},
        # ),
