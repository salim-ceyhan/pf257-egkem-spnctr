"""
Experiment 04 - Information-leakage metrics.

Computes mutual information and local entropy style statistics to assess
residual structure between plaintext and ciphertext.
"""
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
from mpl_toolkits .axes_grid1 import ImageGrid 

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import StatisticalMetrics 


# ======================================================================
# NOTE: (comment removed; repository is English-only)
# ======================================================================
def _to_gray_u8 (img :np .ndarray )->np .ndarray :
    """See README.md for usage and details."""
    x =np .asarray (img )
    if x .ndim ==2 :
        return np .asarray (x ,dtype =np .uint8 ,order ="C")
    if x .ndim ==3 and x .shape [-1 ]in (3 ,4 ):
        g =np .mean (x [...,:3 ],axis =2 )
        return np .asarray (np .clip (np .rint (g ),0 ,255 ),dtype =np .uint8 ,order ="C")
    raise ValueError (f"Unsupported image shape: {x.shape}")


    # ======================================================================
    # Experiment
    # ======================================================================
class PF257EntropyDetailed (BaseExperiment ):
    """
    Experiment 04 (PF257): Mutual Information + Joint/Conditional Entropy + Local Entropy
    """

    def __init__ (self ,config ,cipher ,**_params :Any ):
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="PF257 detailed info-theoretic analysis (MI, joint/conditional, local entropy)",
        )

        # ------------------------------------------------------------------
        # packet["blocks"] -> (Hp,Wp) uint16 field image
        # ------------------------------------------------------------------
    @staticmethod 
    def _blocks4d_to_field2d (blocks :np .ndarray )->np .ndarray :
        if not isinstance (blocks ,np .ndarray )or blocks .ndim !=4 :
            raise ValueError ("packet['blocks'] must be ndarray with shape (ni,nj,bs,bs).")
            # core._blocks_of: Q.reshape(ni,bs,nj,bs).swapaxes(1,2) -> (ni,nj,bs,bs)
            # inverse: blk.swapaxes(1,2).reshape(Hp,Wp)
        ni ,nj ,bs ,_ =blocks .shape 
        out =blocks .swapaxes (1 ,2 ).reshape (ni *bs ,nj *bs )
        return np .asarray (out ,dtype =np .uint16 ,order ="C")

    @staticmethod 
    def _crop (img :np .ndarray ,shape :Tuple [int ,int ])->np .ndarray :
        H ,W =int (shape [0 ]),int (shape [1 ])
        return np .asarray (img [:H ,:W ],order ="C")

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _field_to_u8_view (field2d :np .ndarray )->np .ndarray :
        f =np .asarray (field2d ,dtype =np .uint16 ,order ="C")
        out =np .where (f ==256 ,255 ,f ).astype (np .uint8 ,copy =False )
        return out 

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _entropy_hist (img :np .ndarray ,alphabet :int )->float :
        x =np .asarray (img ).ravel ()
        # NOTE: (comment removed; repository is English-only)
        hist =np .bincount (x .astype (np .int64 ,copy =False ),minlength =alphabet ).astype (np .float64 ,copy =False )
        nz =hist [hist >0 ]
        if nz .size ==0 :
            return 0.0 
        p =nz /float (nz .sum ())
        return float (-np .sum (p *np .log2 (p )))

    @staticmethod 
    def _joint_entropy_bincount (X :np .ndarray ,Y :np .ndarray ,alphabet :int )->float :
        """
        Ortak entropi: H(X,Y)
        - 2D histogram yerine idx = X*alphabet + Y ile tek bincount.
        """
        x =np .asarray (X ).ravel ().astype (np .int64 ,copy =False )
        y =np .asarray (Y ).ravel ().astype (np .int64 ,copy =False )
        if x .size !=y .size :
            raise ValueError ("Joint entropy: size mismatch.")
        idx =x *int (alphabet )+y 
        hist =np .bincount (idx ,minlength =alphabet *alphabet ).astype (np .float64 ,copy =False )
        nz =hist [hist >0 ]
        if nz .size ==0 :
            return 0.0 
        p =nz /float (nz .sum ())
        return float (-np .sum (p *np .log2 (p )))

    def _mutual_information (self ,X :np .ndarray ,Y :np .ndarray ,alphabet :int )->Tuple [float ,float ,float ]:
        """
        Returns: (MI, H_joint, H_cond = H(Y|X))
        MI = H(X) + H(Y) - H(X,Y)
        H(Y|X) = H(X,Y) - H(X)
        """
        if alphabet ==256 :
            Xu =np .asarray (X ,dtype =np .uint8 )
            Yu =np .asarray (Y ,dtype =np .uint8 )
            Hx =float (StatisticalMetrics .entropy (Xu ))
            Hy =float (StatisticalMetrics .entropy (Yu ))
            Hxy =float (self ._joint_entropy_bincount (Xu ,Yu ,256 ))
        else :
            Xp =np .asarray (X ,dtype =np .uint16 )
            Yp =np .asarray (Y ,dtype =np .uint16 )
            Hx =float (self ._entropy_hist (Xp ,alphabet ))
            Hy =float (self ._entropy_hist (Yp ,alphabet ))
            Hxy =float (self ._joint_entropy_bincount (Xp ,Yp ,alphabet ))

        mi =Hx +Hy -Hxy 
        # NOTE: (comment removed; repository is English-only)
        if mi <0 and mi >-1e-10 :
            mi =0.0 
        h_cond =Hxy -Hx 
        return float (mi ),float (Hxy ),float (h_cond )

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _local_entropy_map_u8 (self ,img_u8 :np .ndarray ,block_size :int =16 )->np .ndarray :
        H ,W =img_u8 .shape 
        nb_h =H //block_size 
        nb_w =W //block_size 
        if nb_h ==0 or nb_w ==0 :
            return np .array ([[float (StatisticalMetrics .entropy (img_u8 ))]],dtype =float )

        out =np .zeros ((nb_h ,nb_w ),dtype =float )
        for i in range (nb_h ):
            r0 =i *block_size 
            r1 =r0 +block_size 
            for j in range (nb_w ):
                c0 =j *block_size 
                c1 =c0 +block_size 
                blk =img_u8 [r0 :r1 ,c0 :c1 ]
                out [i ,j ]=float (StatisticalMetrics .entropy (blk ))if blk .size else 0.0 
        return out 

    def _local_entropy_map_field (self ,img_field :np .ndarray ,alphabet :int =257 ,block_size :int =16 )->np .ndarray :
        H ,W =img_field .shape 
        nb_h =H //block_size 
        nb_w =W //block_size 
        if nb_h ==0 or nb_w ==0 :
            return np .array ([[self ._entropy_hist (img_field ,alphabet )]],dtype =float )

        out =np .zeros ((nb_h ,nb_w ),dtype =float )
        for i in range (nb_h ):
            r0 =i *block_size 
            r1 =r0 +block_size 
            for j in range (nb_w ):
                c0 =j *block_size 
                c1 =c0 +block_size 
                blk =img_field [r0 :r1 ,c0 :c1 ]
                out [i ,j ]=self ._entropy_hist (blk ,alphabet )if blk .size else 0.0 
        return out 

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    @staticmethod 
    def _subsample_pair (
    X :np .ndarray ,
    Y :np .ndarray ,
    n :int |None ,
    rng :np .random .Generator ,
    )->tuple [np .ndarray ,np .ndarray ,int ]:
        x =np .asarray (X ).ravel ()
        y =np .asarray (Y ).ravel ()
        if x .size !=y .size :
            raise ValueError ("Subsample: size mismatch.")
        N =x .size 
        if (n is None )or (n >=N ):
            return x ,y ,N 
        idx =rng .choice (N ,size =int (n ),replace =False )
        return x [idx ],y [idx ],int (n )

    def _mi_with_permutation_baseline (
    self ,
    X :np .ndarray ,
    Y :np .ndarray ,
    alphabet :int ,
    *,
    subsample_n :int |None ,
    perm_repeats :int ,
    rng :np .random .Generator ,
    )->dict [str ,float ]:
        """See README.md for usage and details."""
        xs ,ys ,n_used =self ._subsample_pair (X ,Y ,subsample_n ,rng )

        # plug-in MI
        mi ,h_joint ,h_cond =self ._mutual_information (xs ,ys ,alphabet )

        # NOTE: (comment removed; repository is English-only)
        perm_vals =[]
        for _ in range (int (perm_repeats )):
            y_perm =rng .permutation (ys )
            mi_p ,_ ,_ =self ._mutual_information (xs ,y_perm ,alphabet )
            perm_vals .append (mi_p )

        perm_vals =np .asarray (perm_vals ,dtype =np .float64 )
        perm_mean =float (perm_vals .mean ())if perm_vals .size else 0.0 
        perm_std =float (perm_vals .std (ddof =0 ))if perm_vals .size else 0.0 

        mi_excess =mi -perm_mean 
        if mi_excess <0 :
            mi_excess =0.0 

        return {
        "MI":float (mi ),
        "H_joint":float (h_joint ),
        "H_cond":float (h_cond ),
        "MI_perm_mean":float (perm_mean ),
        "MI_perm_std":float (perm_std ),
        "MI_excess":float (mi_excess ),
        "MI_N_used":float (n_used ),
        }

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def _plot_local_entropy_pair (self ,img_name :str ,local_p :np .ndarray ,local_c :np .ndarray ,*,suffix :str )->None :
        fig =plt .figure (figsize =(8 ,4 ))
        grid =ImageGrid (
        fig ,111 ,
        nrows_ncols =(1 ,2 ),
        axes_pad =0.15 ,
        cbar_mode ="single",
        cbar_location ="right",
        cbar_pad =0.15 ,
        )

        im0 =grid [0 ].imshow (local_p ,vmin =0 ,vmax =8 )
        grid [0 ].set_title ("Original (Local H)")
        grid [0 ].axis ("off")

        im1 =grid [1 ].imshow (local_c ,vmin =0 ,vmax =8 )
        grid [1 ].set_title ("Cipher (Local H)")
        grid [1 ].axis ("off")

        grid .cbar_axes [0 ].colorbar (im1 )

        base =img_name .rsplit (".",1 )[0 ]
        out_path =self .output_dir /f"{base}_local_entropy_{suffix}.png"
        plt .savefig (out_path ,dpi =150 ,bbox_inches ="tight",pad_inches =0.05 )
        plt .close (fig )
        self .log (f"  ✓ Local entropy map saved: {out_path.name}")

        # ------------------------------------------------------------------
        # Nonce
        # ------------------------------------------------------------------
    def _nonce64 (self ,img_name :str )->int :
        if bool (getattr (self .config ,"deterministic",False )):
            seed =int (getattr (self .config ,"seed",42 ))
            h =hashlib .sha256 (f"PF257_NONCE|exp04|{seed}|{img_name}".encode ("utf-8")).digest ()
            return int .from_bytes (h [:8 ],"big",signed =False )
        return int (secrets .randbits (64 ))

        # ------------------------------------------------------------------
        # NOTE: (comment removed; repository is English-only)
        # ------------------------------------------------------------------
    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        self .log ("Starting PF257 info-theoretic analysis (MI, joint/conditional, local entropy)...")

        rows :List [Dict [str ,Any ]]=[]
        bs_local =int (getattr (self .config ,"block_size",16 ))

        for img_name ,img_orig in images .items ():
            self .log (f"Processing {img_name}...")

            img_u8 =_to_gray_u8 (img_orig )
            H ,W =img_u8 .shape 
            nonce64 =self ._nonce64 (img_name )

            packet ,out =self .cipher .encrypt (
            img_u8 ,
            nonce =nonce64 ,
            return_priv =False ,
            debug_return_intermediate =False ,
            )

            # Packet shapes
            orig_shape =packet .get ("orig_shape",(H ,W ))
            pad_shape =packet .get ("pad_shape",img_u8 .shape )
            Hp ,Wp =int (pad_shape [0 ]),int (pad_shape [1 ])

            # Cipher field image (Hp,Wp) in 0..256
            blocks4d =packet .get ("blocks",None )
            if blocks4d is None :
            # NOTE: (comment removed; repository is English-only)
                if isinstance (out ,np .ndarray )and out .ndim ==4 :
                    blocks4d =out 
                else :
                    raise ValueError ("packet['blocks'] missing: ciphertext blocks are required for PF257 analysis.")

            blocks4d_u16 =np .asarray (blocks4d ,dtype =np .uint16 )
            cipher_field_full =self ._blocks4d_to_field2d (blocks4d_u16 )

            # NOTE: (comment removed; repository is English-only)
            if cipher_field_full .shape [0 ]<Hp or cipher_field_full .shape [1 ]<Wp :
                raise ValueError (
                f"Cipher blocks unpacked shape {cipher_field_full.shape} is smaller than pad_shape {(Hp, Wp)}."
                )
            cipher_field_full =cipher_field_full [:Hp ,:Wp ]

            cipher_field =self ._crop (cipher_field_full ,orig_shape )# (H,W) uint16
            cipher_u8_view =self ._field_to_u8_view (cipher_field )# (H,W) uint8
            orig_field =np .asarray (self ._crop (img_u8 ,orig_shape ),dtype =np .uint16 )# 0..255 subset

            # ---- Mutual information / entropies ----
            # NOTE: (comment removed; repository is English-only)
            subsample_n =getattr (self .config ,"mi_subsample",200_000 )
            perm_repeats =int (getattr (self .config ,"mi_perm_repeats",10 ))
            seed =int (getattr (self .config ,"seed",42 ))
            seed_offset =int (getattr (self .config ,"mi_perm_seed_offset",12345 ))

            # NOTE: (comment removed; repository is English-only)
            rng =np .random .default_rng (seed +seed_offset +(hash (img_name )&0xFFFFFFFF ))

            # NOTE: (comment removed; repository is English-only)
            r256 =self ._mi_with_permutation_baseline (
            img_u8 ,cipher_u8_view ,256 ,
            subsample_n =subsample_n ,
            perm_repeats =perm_repeats ,
            rng =rng ,
            )
            r257 =self ._mi_with_permutation_baseline (
            orig_field ,cipher_field ,257 ,
            subsample_n =subsample_n ,
            perm_repeats =perm_repeats ,
            rng =rng ,
            )

            mi256 ,hj256 ,hcyx256 =r256 ["MI"],r256 ["H_joint"],r256 ["H_cond"]
            mi257 ,hj257 ,hcyx257 =r257 ["MI"],r257 ["H_joint"],r257 ["H_cond"]


            # ---- Local entropy maps + stats ----
            local_p_256 =self ._local_entropy_map_u8 (img_u8 ,block_size =bs_local )
            local_c_256 =self ._local_entropy_map_u8 (cipher_u8_view ,block_size =bs_local )
            local_c_257 =self ._local_entropy_map_field (cipher_field ,alphabet =257 ,block_size =bs_local )

            rows .append (
            {
            "Image":img_name ,
            "H":int (orig_shape [0 ]),
            "W":int (orig_shape [1 ]),
            "Hp":int (Hp ),
            "Wp":int (Wp ),
            "Nonce64":int (nonce64 ),

            "H_joint_256":float (hj256 ),
            "H_cond_256":float (hcyx256 ),
            "MI_256":float (mi256 ),

            "H_joint_257":float (hj257 ),
            "H_cond_257":float (hcyx257 ),
            "MI_257":float (mi257 ),

            "Local_H_mean_orig_256":float (local_p_256 .mean ()),
            "Local_H_std_orig_256":float (local_p_256 .std ()),
            "Local_H_mean_cview_256":float (local_c_256 .mean ()),
            "Local_H_std_cview_256":float (local_c_256 .std ()),

            "Local_H_mean_cfield_257":float (local_c_257 .mean ()),
            "Local_H_std_cfield_257":float (local_c_257 .std ()),

            "MI_256_perm_mean":float (r256 ["MI_perm_mean"]),
            "MI_256_perm_std":float (r256 ["MI_perm_std"]),
            "MI_256_excess":float (r256 ["MI_excess"]),
            "MI_256_N_used":int (r256 ["MI_N_used"]),

            "MI_257_perm_mean":float (r257 ["MI_perm_mean"]),
            "MI_257_perm_std":float (r257 ["MI_perm_std"]),
            "MI_257_excess":float (r257 ["MI_excess"]),
            "MI_257_N_used":int (r257 ["MI_N_used"]),
            }
            )

            # NOTE: (comment removed; repository is English-only)
            self ._plot_local_entropy_pair (img_name ,local_p_256 ,local_c_256 ,suffix ="256")

            self .log (
            f"  MI256={mi256:.6f} (perm={r256['MI_perm_mean']:.6f}±{r256['MI_perm_std']:.6f}, "
            f"excess={r256['MI_excess']:.6f}, N={int(r256['MI_N_used'])}) | "
            f"MI257={mi257:.6f} (perm={r257['MI_perm_mean']:.6f}±{r257['MI_perm_std']:.6f}, "
            f"excess={r257['MI_excess']:.6f}, N={int(r257['MI_N_used'])})"
            )


        df =pd .DataFrame (rows )
        self .save_results (df ,"entropy_detailed_pf257.xlsx")

        # NOTE: (comment removed; repository is English-only)
        try :
            self .export_latex (
            df .copy (),
            caption ="Mutual information and local entropy statistics for PF257-EGKEM-SPNCTR.",
            label ="tab:mi_local_pf257",
            )
        except Exception :
            pass 

        self .log ("PF257 info-theoretic analysis completed.")
        return df 
