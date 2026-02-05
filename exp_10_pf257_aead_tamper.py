# exp_10_aead_tamper.py
# ---------------------------------------------------------------------
# Experiment 10 — AEAD / IND-CCA Tamper Suite (PF257-EGKEM-SPNCTR uyumlu)
# ---------------------------------------------------------------------
from __future__ import annotations 

from typing import Any ,Dict ,List ,Tuple ,Callable ,cast 
import sys 
from pathlib import Path 
import zlib 
import inspect 

import numpy as np 
import pandas as pd 

from pf257_experiment_base import BaseExperiment 


class PF257AEADTamperSuite (BaseExperiment ):
    """AEAD/tamper resistance checks (Experiment 10)."""

    # NOTE: (comment removed; repository is English-only)
    N_BITFLIP :int =1024 
    N_AD :int =512 
    N_STRUCT :int =256 
    N_CLEAN :int =64 
    N_WRONGK :int =128 

    def __init__ (self ,config ,cipher ,**params :Any )->None :
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="(PF257) AEAD tamper + wrong-key (IND-CCA) suite",
        )

        self .N_BITFLIP =int (params .get ("n_bitflip",self .N_BITFLIP ))
        self .N_AD =int (params .get ("n_ad",self .N_AD ))
        self .N_STRUCT =int (params .get ("n_struct",self .N_STRUCT ))
        self .N_CLEAN =int (params .get ("n_clean",self .N_CLEAN ))
        self .N_WRONGK =int (params .get ("n_wrongk",self .N_WRONGK ))

        # ----------------------------- stats helpers -----------------------------

    @staticmethod 
    def _rng (seed :int )->np .random .Generator :
        return np .random .default_rng (int (seed )&0xFFFFFFFF )

    @staticmethod 
    def _wilson_upper (k :int ,n :int ,z :float )->float :
        if n <=0 :
            return float ("nan")
        nn =float (n )
        phat =float (k )/nn 
        denom =1.0 +(z *z )/nn 
        center =(phat +(z *z )/(2.0 *nn ))/denom 
        rad =z *np .sqrt ((phat *(1.0 -phat )/nn )+(z *z )/(4.0 *nn *nn ))/denom 
        return float (center +rad )

        # ----------------------------- packet helpers -----------------------------

    @staticmethod 
    def _deepcopy_packet (pkt :Dict [str ,Any ])->Dict [str ,Any ]:
        out :Dict [str ,Any ]={}
        for k ,v in pkt .items ():
            out [k ]=v .copy ()if isinstance (v ,np .ndarray )else v 
        return out 

    def _decrypt_verify (self ,packet :Dict [str ,Any ],bob_priv :Any ,bob_pub_raw :Any |None =None )->np .ndarray :
        dec =cast (Any ,self .cipher ).decrypt 
        try :
            sig =inspect .signature (dec )
            params =set (sig .parameters .keys ())
        except Exception :
            params =set ()

        kwargs :Dict [str ,Any ]={"verify":True }

        if "bob_priv"in params :
            kwargs ["bob_priv"]=bob_priv 
        elif "bob_priv_x"in params :
            kwargs ["bob_priv_x"]=bob_priv 
        else :
            return dec (packet ,bob_priv ,True )# type: ignore[misc]

        if bob_pub_raw is not None and ("bob_pub_raw"in params or not params ):
            kwargs ["bob_pub_raw"]=bob_pub_raw 

        return dec (packet ,**kwargs )

    def _judge_decrypt (self ,packet :Dict [str ,Any ],bob_priv :Any ,original_img :np .ndarray )->Tuple [bool ,bool ,str ]:
        try :
            bob_pub_raw =getattr (self ,"_bob_pub_raw_current",None )
            dec_img =self ._decrypt_verify (packet ,bob_priv ,bob_pub_raw =bob_pub_raw )
            ok_exact =bool (np .array_equal (dec_img ,original_img ))
            if ok_exact :
                return True ,True ,"accepted"
            return True ,False ,"false-return"
        except Exception :
            return False ,False ,"exception"

            # ----------------------------- tamper primitives -----------------------------

    @staticmethod 
    def _tamper_bitflip_blocks (pkt :Dict [str ,Any ],n_changes :int ,rng :np .random .Generator )->Dict [str ,Any ]:
        tp =PF257AEADTamperSuite ._deepcopy_packet (pkt )
        blocks =cast (np .ndarray ,tp .get ("blocks",None ))
        if blocks is None or not isinstance (blocks ,np .ndarray ):
            return tp 

        flat =blocks .reshape (-1 )
        n =int (max (1 ,n_changes ))
        n =int (min (n ,flat .size ))

        idx =rng .choice (flat .size ,size =n ,replace =False )
        delta =rng .integers (1 ,257 ,size =n ,dtype =np .uint16 )# 1..256
        x =flat [idx ].astype (np .uint16 ,copy =False )
        flat [idx ]=((x +delta )%257 ).astype (np .uint16 ,copy =False )

        tp ["blocks"]=blocks 
        return tp 

    @staticmethod 
    def _tamper_ad_field (pkt :Dict [str ,Any ],field :str ,rng :np .random .Generator )->Tuple [Dict [str ,Any ],bool ]:
    # NOTE: (comment removed; repository is English-only)
        if field !="tag"and field not in pkt :
            return PF257AEADTamperSuite ._deepcopy_packet (pkt ),True 

        tp =PF257AEADTamperSuite ._deepcopy_packet (pkt )

        if field =="nonce":
            tp ["nonce"]=int (tp ["nonce"])^int (rng .integers (1 ,(1 <<63 )-1 ,dtype =np .int64 ))

        elif field =="salt":
            s =bytearray (cast (bytes ,tp ["salt"]))
            i =int (rng .integers (0 ,len (s )))
            s [i ]^=int (rng .integers (1 ,255 ))
            tp ["salt"]=bytes (s )

        elif field =="kem_epk":
            b =bytearray (cast (bytes ,tp ["kem_epk"]))
            b [0 ]^=0xA5 
            tp ["kem_epk"]=bytes (b )

        elif field =="bs":
            tp ["bs"]=int (tp ["bs"])+1 

        elif field =="orig_shape":
            H ,W =cast (Tuple [int ,int ],tp ["orig_shape"])
            tp ["orig_shape"]=(max (1 ,int (H )-1 ),int (W ))

        elif field =="pad_shape":
            Hp ,Wp =cast (Tuple [int ,int ],tp ["pad_shape"])
            tp ["pad_shape"]=(int (Hp ),max (1 ,int (Wp )-1 ))

        elif field =="enc_params":
            b =bytearray (cast (bytes ,tp ["enc_params"]))
            j =int (rng .integers (0 ,len (b )))
            b [j ]^=0x80 
            tp ["enc_params"]=bytes (b )

        elif field =="tag":
            t0 =tp .get ("tag",None )
            if not isinstance (t0 ,(bytes ,bytearray ))or len (t0 )==0 :
                return tp ,True 
            t =bytearray (cast (bytes ,t0 ))
            j =int (rng .integers (0 ,len (t )))
            t [j ]^=0x40 
            tp ["tag"]=bytes (t )

        else :
            return tp ,True 

        return tp ,False 

    @staticmethod 
    def _tamper_permute_blocks (pkt :Dict [str ,Any ],rng :np .random .Generator )->Tuple [Dict [str ,Any ],bool ]:
        tp =PF257AEADTamperSuite ._deepcopy_packet (pkt )
        blocks =cast (np .ndarray ,tp .get ("blocks",None ))
        if blocks is None or not isinstance (blocks ,np .ndarray )or blocks .ndim !=4 :
            return tp ,True 

        bh ,bw ,_ ,_ =blocks .shape 
        if bh ==1 and bw ==1 :
            return tp ,True 

        if bh >1 :
            i ,j =rng .choice (bh ,size =2 ,replace =False )
            blocks [[i ,j ],:,:,:]=blocks [[j ,i ],:,:,:]

        if bw >1 :
            i ,j =rng .choice (bw ,size =2 ,replace =False )
            blocks [:,[i ,j ],:,:]=blocks [:,[j ,i ],:,:]

        tp ["blocks"]=blocks 
        return tp ,False 

    @staticmethod 
    def _tamper_truncate_blocks (pkt :Dict [str ,Any ])->Tuple [Dict [str ,Any ],bool ]:
        tp =PF257AEADTamperSuite ._deepcopy_packet (pkt )
        blocks =cast (np .ndarray ,tp .get ("blocks",None ))
        if blocks is None or not isinstance (blocks ,np .ndarray )or blocks .ndim !=4 :
            return tp ,True 
        bh ,bw ,_ ,_ =blocks .shape 
        if bw >1 :
            tp ["blocks"]=blocks [:,:bw -1 ,:,:]
            return tp ,False 
        if bh >1 :
            tp ["blocks"]=blocks [:bh -1 ,:,:,:]
            return tp ,False 
        return tp ,True 

    @staticmethod 
    def _tamper_append_blocks (pkt :Dict [str ,Any ],rng :np .random .Generator )->Tuple [Dict [str ,Any ],bool ]:
        tp =PF257AEADTamperSuite ._deepcopy_packet (pkt )
        blocks =cast (np .ndarray ,tp .get ("blocks",None ))
        if blocks is None or not isinstance (blocks ,np .ndarray )or blocks .ndim !=4 :
            return tp ,True 
        bh ,_bw ,bs1 ,bs2 =blocks .shape 
        extra =rng .integers (0 ,257 ,size =(bh ,1 ,bs1 ,bs2 ),dtype =np .uint16 )
        tp ["blocks"]=np .concatenate ([blocks ,extra ],axis =1 )
        return tp ,False 

    @staticmethod 
    def _make_wrong_bob_priv (bob_priv :Any ,rng :np .random .Generator )->Any :
    # NOTE: (comment removed; repository is English-only)
        try :
            from cryptography .hazmat .primitives .asymmetric import x25519 
            if isinstance (bob_priv ,x25519 .X25519PrivateKey ):
                return x25519 .X25519PrivateKey .generate ()
        except Exception :
            pass 
        return rng .integers (0 ,256 ,32 ,dtype =np .uint8 ).tobytes ()

        # ----------------------------- saving -----------------------------

    def _save_xlsx (self ,df :pd .DataFrame ,filename :str )->None :
        if hasattr (self ,"save_results")and callable (getattr (self ,"save_results")):
            self .save_results (df ,filename )
            return 
        df .to_excel (filename ,index =False )

        # ------------------------------- run -------------------------------

    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("AEAD tamper suite (PF257) started...")

        base_seed =int (getattr (self .config ,"seed",0 ))
        rng =self ._rng (base_seed )

        rows :List [Dict [str ,Any ]]=[]

        def append_row (
        img_name :str ,
        shape_str :str ,
        nonce64 :int ,
        case :str ,
        case_class :str ,
        trials :int ,
        accept_any :int ,
        accept_exact :int ,
        rej_exc :int ,
        rej_false :int ,
        skipped :bool ,
        )->None :
            far_any =100.0 *accept_any /float (trials )if trials >0 else float ("nan")
            far_exact =100.0 *accept_exact /float (trials )if trials >0 else float ("nan")
            wil99 =100.0 *self ._wilson_upper (accept_any ,trials ,z =2.5758293035489004 )if trials >0 else float ("nan")

            rows .append ({
            "Image":img_name ,
            "Shape":shape_str ,
            "Nonce64":int (nonce64 ),
            "Case":case ,
            "CaseClass":case_class ,
            "Trials":int (trials ),
            "Accept_any":int (accept_any ),
            "FAR_any_%":float (far_any ),
            "FAR_any_Wilson99_UL_%":float (wil99 ),
            "Accept_exact":int (accept_exact ),
            "FAR_exact_%":float (far_exact ),
            "Reject_exception":int (rej_exc ),
            "Reject_false_return":int (rej_false ),
            "Skipped":bool (skipped ),
            })

        for img_name ,img in images .items ():
            self ._bob_pub_raw_current =None # type: ignore[attr-defined]

            img_u8 =np .asarray (img ,dtype =np .uint8 ,order ="C")
            shape_str =f"{int(img_u8.shape[0])}x{int(img_u8.shape[1])}"
            nonce64 =int (rng .integers (0 ,2 **64 ,dtype =np .uint64 ))

            enc_res =self .cipher .encrypt (img_u8 ,nonce =nonce64 ,return_priv =True )
            if not isinstance (enc_res ,tuple ):
                raise RuntimeError ("See README.md for usage and details.")
            if len (enc_res )==3 :
                packet ,bob_priv ,bob_pub_raw =enc_res 
                self ._bob_pub_raw_current =bob_pub_raw # type: ignore[attr-defined]
            elif len (enc_res )==2 :
                packet ,bob_priv =enc_res 
                self ._bob_pub_raw_current =None # type: ignore[attr-defined]
            else :
                raise RuntimeError (f"See README.md for usage and details.")

            packet =cast (Dict [str ,Any ],packet )

            ok_any ,ok_exact ,_ =self ._judge_decrypt (packet ,bob_priv ,img_u8 )
            if not (ok_any and ok_exact ):
                raise RuntimeError (f"Clean decrypt failed for {img_name} (verify=True).")

            def run_case (case :str ,case_class :str ,trials :int ,maker :Callable [[],Tuple [Dict [str ,Any ],bool ]])->None :
                if trials <=0 :
                    append_row (img_name ,shape_str ,nonce64 ,case ,case_class ,0 ,0 ,0 ,0 ,0 ,True )
                    return 

                acc_any =0 
                acc_exact =0 
                rej_exc =0 
                rej_false =0 
                skipped =False 

                for _ in range (int (trials )):
                    pkt2 ,sk =maker ()
                    if sk :
                        skipped =True 
                        continue 
                    ok_a ,ok_e ,_mode =self ._judge_decrypt (pkt2 ,bob_priv ,img_u8 )
                    if ok_a :
                        acc_any +=1 
                        if ok_e :
                            acc_exact +=1 
                        else :
                            rej_false +=1 
                    else :
                        rej_exc +=1 

                append_row (img_name ,shape_str ,nonce64 ,case ,case_class ,int (trials ),acc_any ,acc_exact ,rej_exc ,rej_false ,skipped )

            run_case ("clean","control",self .N_CLEAN ,lambda :(packet ,False ))

            blocks0 =packet .get ("blocks",None )
            if isinstance (blocks0 ,np .ndarray ):
                n_total =int (blocks0 .size )
                run_case ("bitflip_min","bitflip",self .N_BITFLIP ,lambda :(self ._tamper_bitflip_blocks (packet ,1 ,rng ),False ))
                run_case ("bitflip_1pct","bitflip",self .N_BITFLIP ,lambda :(self ._tamper_bitflip_blocks (packet ,max (1 ,int (0.01 *n_total )),rng ),False ))
                run_case ("bitflip_10pct","bitflip",self .N_BITFLIP ,lambda :(self ._tamper_bitflip_blocks (packet ,max (1 ,int (0.10 *n_total )),rng ),False ))
            else :
                run_case ("bitflip_min","bitflip",0 ,lambda :(packet ,True ))
                run_case ("bitflip_1pct","bitflip",0 ,lambda :(packet ,True ))
                run_case ("bitflip_10pct","bitflip",0 ,lambda :(packet ,True ))

            for f in ("nonce","salt","kem_epk","bs","orig_shape","pad_shape","enc_params","tag"):
                run_case (f"AD_{f}","AD",self .N_AD ,lambda ff =f :self ._tamper_ad_field (packet ,ff ,rng ))

            run_case ("permute_blocks","structure",self .N_STRUCT ,lambda :self ._tamper_permute_blocks (packet ,rng ))
            run_case ("truncate_blocks","structure",self .N_STRUCT ,lambda :self ._tamper_truncate_blocks (packet ))
            run_case ("append_blocks","structure",self .N_STRUCT ,lambda :self ._tamper_append_blocks (packet ,rng ))

            wrong_rng =self ._rng (base_seed ^(zlib .crc32 (img_name .encode ("utf-8"))&0xFFFFFFFF )^0xC0FFEE )
            acc_any =0 
            acc_exact =0 
            rej_exc =0 
            rej_false =0 
            for _ in range (self .N_WRONGK ):
                wrong_priv =self ._make_wrong_bob_priv (bob_priv ,wrong_rng )
                ok_a ,ok_e ,_mode =self ._judge_decrypt (packet ,wrong_priv ,img_u8 )
                if ok_a :
                    acc_any +=1 
                    if ok_e :
                        acc_exact +=1 
                    else :
                        rej_false +=1 
                else :
                    rej_exc +=1 
            append_row (img_name ,shape_str ,nonce64 ,"wrong_key","keying",self .N_WRONGK ,acc_any ,acc_exact ,rej_exc ,rej_false ,False )

        df =pd .DataFrame (rows )
        self ._save_xlsx (df ,"aead_tamper_pf257_rows.xlsx")

        df2 =df [(df ["Trials"]>0 )&(~df ["Skipped"])].copy ()
        if not df2 .empty :
            summary =(
            df2 .groupby (["Case","CaseClass"],as_index =False )[
            ["Trials","Accept_any","Accept_exact","Reject_exception","Reject_false_return"]
            ].sum ()
            )
            summary ["FAR_any_%"]=100.0 *summary ["Accept_any"]/summary ["Trials"].astype (float )
            summary ["FAR_exact_%"]=100.0 *summary ["Accept_exact"]/summary ["Trials"].astype (float )
            summary ["FAR_any_Wilson99_UL_%"]=[
            100.0 *self ._wilson_upper (int (k ),int (n ),z =2.5758293035489004 )
            for k ,n in zip (summary ["Accept_any"].tolist (),summary ["Trials"].tolist ())
            ]
            self ._save_xlsx (summary ,"aead_tamper_pf257_summary.xlsx")

            if hasattr (self ,"export_latex")and callable (getattr (self ,"export_latex")):
                self .export_latex (
                summary ,
                caption ="See README.md for usage and details.",
                label ="tab:aead-tamper-pf257",
                )

        return df 
