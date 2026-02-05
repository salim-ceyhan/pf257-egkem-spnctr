"""
Experiment 05 - Keystream/mask randomness tests (NIST-inspired).

Generates keystream/mask bitstreams and runs a small NIST-inspired battery,
reporting per-test p-values and aggregate summaries.
"""
from __future__ import annotations 

from typing import Dict ,List ,Tuple ,Any ,Callable ,cast ,Optional 
import sys 
from pathlib import Path 
import math 
import hashlib 
import time 

import numpy as np 
import pandas as pd 
import matplotlib .pyplot as plt 

from pf257_experiment_base import BaseExperiment 
from pf257_experiment_framework import ExperimentConfig 


class PF257RandomnessTests (BaseExperiment ):
    """Keystream/mask randomness tests (NIST-inspired) (Experiment 05)."""

    # NOTE: (comment removed; repository is English-only)
    DEFAULT_N_STREAMS :int =200 
    DEFAULT_BITS_PER_STREAM :int =1_000_000 
    DEFAULT_BLOCK_FREQ_M :int =128 
    DEFAULT_ALPHA :float =0.01 
    DEFAULT_UNI_BINS :int =10 
    DEFAULT_P_UNI_THRESH :float =1e-4 
    DEFAULT_SOURCE :str ="bytes"# "bytes" | "field9"

    # Pratik cap'ler
    MAXBITS_MONOBIT :int =1_000_000 
    MAXBITS_BLOCKFREQ :int =1_000_000 
    MAXBITS_RUNS :int =1_000_000 
    MAXBITS_CUSUM :int =1_000_000 
    MAXBITS_LONGESTRUN :int =200_000 
    MAXBITS_FFT :int =131_072 
    MAXBITS_LINCOMP :int =100_000 

    def __init__ (self ,config :ExperimentConfig ,cipher ,**params :Any ):
        module =sys .modules .get (self .__class__ .__module__ )
        module_path =getattr (module ,"__file__",self .__class__ .__name__ )
        base_name =Path (module_path ).stem 
        clean_name =base_name .removeprefix ("experiment_").removeprefix ("exp_")

        super ().__init__ (
        config =config ,
        cipher =cipher ,
        name =clean_name ,
        description ="PF257 stream randomness (NIST-inspired + second-level checks)",
        )

        self .N_STREAMS =int (params .get ("n_streams",self .DEFAULT_N_STREAMS ))
        self .BITS_PER_STREAM =int (params .get ("bits_per_stream",self .DEFAULT_BITS_PER_STREAM ))
        self .BLOCK_FREQ_M =int (params .get ("block_freq_m",self .DEFAULT_BLOCK_FREQ_M ))
        self .ALPHA =float (params .get ("alpha",self .DEFAULT_ALPHA ))
        self .UNI_BINS =int (params .get ("uniformity_bins",self .DEFAULT_UNI_BINS ))
        self .P_UNI_THRESH =float (params .get ("p_uniform_thresh",self .DEFAULT_P_UNI_THRESH ))
        self .SOURCE =str (params .get ("source",self .DEFAULT_SOURCE )).strip ().lower ()

        self .ENABLE_LONGESTRUN =bool (params .get ("enable_longestrun",True ))
        self .ENABLE_FFT =bool (params .get ("enable_fft",True ))
        self .ENABLE_LINCOMP =bool (params .get ("enable_lincomp",False ))

        self ._cap_override :Dict [str ,int ]={
        "monobit":int (params .get ("cap_monobit",self .MAXBITS_MONOBIT )),
        "blockfreq":int (params .get ("cap_blockfreq",self .MAXBITS_BLOCKFREQ )),
        "runs":int (params .get ("cap_runs",self .MAXBITS_RUNS )),
        "cusum":int (params .get ("cap_cusum",self .MAXBITS_CUSUM )),
        "longestrun":int (params .get ("cap_longestrun",self .MAXBITS_LONGESTRUN )),
        "fft":int (params .get ("cap_fft",self .MAXBITS_FFT )),
        "lincomp":int (params .get ("cap_lincomp",self .MAXBITS_LINCOMP )),
        }

        if self .SOURCE not in ("bytes","field9"):
            raise ValueError ("source must be 'bytes' or 'field9'.")

            # NOTE: (comment removed; repository is English-only)

    def _get_pf257_core (self ):
        cw =self .cipher 
        adapter =getattr (cw ,"cipher",None )
        if adapter is None :
            raise AttributeError ("See README.md for usage and details.")
        core =getattr (adapter ,"core",None )
        return core if core is not None else adapter 

        # ---------------- Deterministik key/nonce ----------------

    @staticmethod 
    def _derive_key32 (seed :int )->bytes :
        return hashlib .sha256 (f"PF257_NIST_KEY|{int(seed)}".encode ("utf-8")).digest ()

    @staticmethod 
    def _derive_nonce63 (seed :int ,i :int )->int :
    # deterministik + benzersiz (pratikte)
        h =hashlib .sha256 (f"PF257_NIST_NONCE|{int(seed)}|{int(i)}".encode ("utf-8")).digest ()
        x =int .from_bytes (h [:8 ],"big",signed =False )
        return int (x &((1 <<63 )-1 ))

        # NOTE: (comment removed; repository is English-only)

    def _bits_from_bytes_stream (self ,n_bits :int ,key32 :bytes ,nonce :int )->np .ndarray :
        core =self ._get_pf257_core ()
        fn =getattr (core ,"_bytes_stream",None )
        if not callable (fn ):
            raise RuntimeError ("See README.md for usage and details.")
        n_bytes =(int (n_bits )+7 )//8 
        ks_bytes =cast (bytes ,fn (key32 ,n_bytes ,int (nonce )))
        bits =np .unpackbits (np .frombuffer (ks_bytes ,dtype =np .uint8 ))
        return bits [:int (n_bits )].astype (np .uint8 ,copy =False )

    def _bits_from_field_stream_9bit (self ,n_bits :int ,key32 :bytes ,nonce :int )->np .ndarray :
        """See README.md for usage and details."""
        core =self ._get_pf257_core ()
        fn =getattr (core ,"_field_stream",None )
        if not callable (fn ):
            raise RuntimeError ("See README.md for usage and details.")

            # NOTE: (comment removed; repository is English-only)
        n_vals =(int (n_bits )+8 )//9 
        vals =cast (np .ndarray ,fn (key32 ,int (n_vals ),int (nonce )))
        vals_u16 =np .asarray (vals ,dtype =np .uint16 )

        # uint16 big-endian bytes -> unpackbits -> her 16-bit grubun son 9 biti
        raw =vals_u16 .astype (">u2",copy =False ).tobytes ()
        b16 =np .unpackbits (np .frombuffer (raw ,dtype =np .uint8 )).reshape (-1 ,16 )
        b9 =b16 [:,7 :]# son 9 bit
        out =b9 .reshape (-1 ).astype (np .uint8 ,copy =False )
        return out [:int (n_bits )]

    def _keystream_bits (self ,n_bits :int ,key32 :bytes ,nonce :int )->np .ndarray :
        if self .SOURCE =="bytes":
            return self ._bits_from_bytes_stream (n_bits ,key32 ,nonce )
        return self ._bits_from_field_stream_9bit (n_bits ,key32 ,nonce )

        # ---------------- χ² sf (SciPy yok) ----------------

    @staticmethod 
    def _gammaincc (a :float ,x :float )->float :
        """See README.md for usage and details."""
        if x <=0 :
            return 1.0 
        if a <=0 :
            return float ("nan")

        EPS =1e-12 
        ITMAX =400 

        def _gser (a_ :float ,x_ :float )->float :
            gln =math .lgamma (a_ )
            ap =a_ 
            s =1.0 /a_ 
            d =s 
            for _ in range (ITMAX ):
                ap +=1.0 
                d *=x_ /ap 
                s +=d 
                if abs (d )<abs (s )*EPS :
                    break 
            return s *math .exp (-x_ +a_ *math .log (x_ )-gln )# P(a,x)

        def _gcf (a_ :float ,x_ :float )->float :
            gln =math .lgamma (a_ )
            b =x_ +1.0 -a_ 
            c =1.0 /1e-30 
            d =1.0 /b 
            h =d 
            for i in range (1 ,ITMAX +1 ):
                an =-i *(i -a_ )
                b +=2.0 
                d =an *d +b 
                if abs (d )<1e-30 :
                    d =1e-30 
                c =b +an /c 
                if abs (c )<1e-30 :
                    c =1e-30 
                d =1.0 /d 
                delta =d *c 
                h *=delta 
                if abs (delta -1.0 )<EPS :
                    break 
            return math .exp (-x_ +a_ *math .log (x_ )-gln )*h # Q(a,x)

        if x <a +1.0 :
            p =_gser (a ,x )
            q =1.0 -p 
            return float (max (0.0 ,min (1.0 ,q )))
        q =_gcf (a ,x )
        return float (max (0.0 ,min (1.0 ,q )))

    @staticmethod 
    def _norm_sf (z :float )->float :
        """Normal(0,1) survival: P[Z >= z]."""
        return 0.5 *math .erfc (z /math .sqrt (2.0 ))

    def _chi2_sf (self ,chi2 :float ,df :int )->float :
        """See README.md for usage and details."""
        if df <=0 or chi2 <0 :
            return float ("nan")

        dff =float (df )
        x =float (chi2 )

        # NOTE: (comment removed; repository is English-only)
        if df <=200 :
            return float (self ._gammaincc (0.5 *dff ,0.5 *x ))

            # NOTE: (comment removed; repository is English-only)
            # z ≈ ((x/df)^(1/3) - (1 - 2/(9df))) / sqrt(2/(9df))
        t =(x /dff )**(1.0 /3.0 )
        mu =1.0 -2.0 /(9.0 *dff )
        sig =math .sqrt (2.0 /(9.0 *dff ))
        z =(t -mu )/sig 
        return float (max (0.0 ,min (1.0 ,self ._norm_sf (z ))))

        # ---------------- NIST second-level ----------------

    @staticmethod 
    def _nist_proportion_ci (m :int ,alpha :float )->Tuple [float ,float ]:
        if m <=0 :
            return float ("nan"),float ("nan")
        p0 =1.0 -float (alpha )
        delta =3.0 *math .sqrt (p0 *(1.0 -p0 )/float (m ))
        return float (p0 -delta ),float (p0 +delta )

    def _nist_pvalue_uniformity (self ,pvals :np .ndarray ,bins :int =10 )->Tuple [float ,float ,np .ndarray ]:
        pvals =np .asarray (pvals ,dtype =float )
        pvals =pvals [np .isfinite (pvals )]
        m =int (pvals .size )
        if m ==0 :
            return float ("nan"),float ("nan"),np .zeros (bins ,dtype =np .int64 )

        hist ,_ =np .histogram (pvals ,bins =bins ,range =(0.0 ,1.0 ))
        expected =m /float (bins )
        chi2 =float (np .sum ((hist -expected )**2 /(expected +1e-12 )))
        p_uniform =float (self ._chi2_sf (chi2 ,df =bins -1 ))
        return chi2 ,p_uniform ,hist .astype (np .int64 )

    def _save_uniformity_plot (self ,test_name :str ,hist :np .ndarray )->None :
        out_dir =getattr (self ,"output_dir",None )
        if out_dir is None :
        # NOTE: (comment removed; repository is English-only)
            out_dir =Path (".")
        else :
            out_dir =Path (out_dir )

        pdir =out_dir /"pvalue_uniformity"
        pdir .mkdir (parents =True ,exist_ok =True )

        fig ,ax =plt .subplots (figsize =(6 ,3 ))
        x =np .arange (hist .size )
        ax .bar (x ,hist )
        ax .set_title (f"p-value histogram (bins={hist.size}) — {test_name}")
        ax .set_xlabel ("Bin index")
        ax .set_ylabel ("Count")
        plt .tight_layout ()
        out_path =pdir /f"pvals_hist_{test_name.replace('/', '_').replace(' ', '_')}.png"
        plt .savefig (out_path ,dpi =150 )
        plt .close (fig )

        # NOTE: (comment removed; repository is English-only)

    @staticmethod 
    def _p_monobit (bits :np .ndarray )->float :
        n =int (bits .size )
        if n ==0 :
            return float ("nan")
        s =float (np .sum (2 *bits .astype (np .int32 )-1 ))
        return float (math .erfc (abs (s )/math .sqrt (2.0 *n )))

    def _p_block_freq (self ,bits :np .ndarray ,M :int )->float :
        n =int (bits .size )
        if n <M :
            return float ("nan")
        N =n //M 
        blocks =bits [:N *M ].reshape (N ,M )
        pi =blocks .mean (axis =1 )
        chi2 =4.0 *M *float (np .sum ((pi -0.5 )**2 ))
        return float (self ._chi2_sf (chi2 ,df =int (N )))

    @staticmethod 
    def _p_runs (bits :np .ndarray )->float :
        n =int (bits .size )
        if n <2 :
            return float ("nan")
        pi =float (bits .mean ())
        if abs (pi -0.5 )>=2.0 /math .sqrt (n ):
            return 0.0 
        v =1 +float (np .sum (bits [1 :]!=bits [:-1 ]))
        num =abs (v -2.0 *n *pi *(1.0 -pi ))
        den =2.0 *math .sqrt (2.0 *n )*pi *(1.0 -pi )
        return float (math .erfc (num /den ))

    def _p_longest_run (self ,bits :np .ndarray )->float :
        n =int (bits .size )
        if n <10_000 :
            return float ("nan")

        M =10_000 
        N =n //M 
        blocks =bits [:N *M ].reshape (N ,M )

        def longest_run_ones (x :np .ndarray )->int :
            mx =0 
            run =0 
            for b in x :
                if b ==1 :
                    run +=1 
                    if run >mx :
                        mx =run 
                else :
                    run =0 
            return mx 

        L =np .array ([longest_run_ones (blocks [i ])for i in range (N )],dtype =np .int32 )

        # kategoriler: <=10, 11, 12, 13, 14, 15, >=16
        counts =np .zeros (7 ,dtype =np .int32 )
        counts [0 ]=int (np .sum (L <=10 ))
        counts [1 ]=int (np .sum (L ==11 ))
        counts [2 ]=int (np .sum (L ==12 ))
        counts [3 ]=int (np .sum (L ==13 ))
        counts [4 ]=int (np .sum (L ==14 ))
        counts [5 ]=int (np .sum (L ==15 ))
        counts [6 ]=int (np .sum (L >=16 ))

        pi =np .array ([0.0882 ,0.2092 ,0.2483 ,0.1933 ,0.1208 ,0.0675 ,0.0727 ],dtype =np .float64 )
        expected =float (N )*pi 
        chi2 =float (np .sum ((counts -expected )**2 /(expected +1e-12 )))
        return float (self ._chi2_sf (chi2 ,df =6 ))

    @staticmethod 
    def _p_fft (bits :np .ndarray )->float :
        n =int (bits .size )
        if n <2 :
            return float ("nan")
        x =2 *bits .astype (np .int32 )-1 
        s =np .fft .rfft (x )
        mag =np .abs (s )[1 :]# NOTE: (comment removed; repository is English-only)
        T =math .sqrt (math .log (1.0 /0.05 )*n )
        N0 =0.95 *(n /2.0 )
        N1 =float (np .sum (mag <T ))
        d =(N1 -N0 )/math .sqrt (n *0.95 *0.05 /4.0 )
        return float (math .erfc (abs (d )/math .sqrt (2.0 )))

    @staticmethod 
    def _p_cusum (bits :np .ndarray ,forward :bool =True )->float :
        n =int (bits .size )
        if n ==0 :
            return float ("nan")
        x =2 *bits .astype (np .int32 )-1 
        if not forward :
            x =x [::-1 ]
        s =np .cumsum (x )
        z =float (np .max (np .abs (s )))
        sqrt_n =math .sqrt (n )

        def phi (t :float )->float :
            return 0.5 *math .erfc (-t /math .sqrt (2.0 ))

        start_k1 =math .floor ((-n /z +1.0 )/4.0 )
        end_k1 =math .floor ((n /z -1.0 )/4.0 )
        start_k2 =math .floor ((-n /z -3.0 )/4.0 )
        end_k2 =math .floor ((n /z -1.0 )/4.0 )

        sum1 =0.0 
        for k in range (int (start_k1 ),int (end_k1 )+1 ):
            sum1 +=phi ((4 *k +1 )*z /sqrt_n )-phi ((4 *k -1 )*z /sqrt_n )

        sum2 =0.0 
        for k in range (int (start_k2 ),int (end_k2 )+1 ):
            sum2 +=phi ((4 *k +3 )*z /sqrt_n )-phi ((4 *k +1 )*z /sqrt_n )

        p_value =1.0 -sum1 +sum2 
        return float (max (0.0 ,min (1.0 ,p_value )))

    @staticmethod 
    def _linear_complexity_BM (bits :np .ndarray ,M :int =500 )->float :
        n =int (bits .size )
        K =n //M 
        if K ==0 :
            return float ("nan")

        Ls :List [int ]=[]
        for k in range (K ):
            s =bits [k *M :(k +1 )*M ].astype (np .int8 )
            c =np .zeros (M ,dtype =np .int8 )
            b =np .zeros (M ,dtype =np .int8 )
            c [0 ]=1 
            b [0 ]=1 
            L =0 
            m =-1 
            for N in range (M ):
                d =s [N ]
                for i in range (1 ,L +1 ):
                    d ^=(c [i ]&s [N -i ])
                if d ==1 :
                    t =c .copy ()
                    p =np .zeros (M ,dtype =np .int8 )
                    shift =N -m 
                    if shift <M :
                        p [shift :]=b [:M -shift ]
                    c =c ^p 
                    if 2 *L <=N :
                        L =N +1 -L 
                        b =t 
                        m =N 
            Ls .append (int (L ))

        L_bar =float (np .mean (Ls ))
        mu =M /2.0 +(9 +(-1 )**M )/36.0 -(M /3.0 +2 /9.0 )/(2 **M )
        sigma2 =(M *(M +2 ))/45.0 
        sigma =math .sqrt (sigma2 )if sigma2 >0 else 1.0 
        T =(L_bar -mu )/sigma 
        return float (math .erfc (abs (T )/math .sqrt (2.0 )))

        # ---------------- Cap helper ----------------

    def _cap_for_test (self ,name :str )->int :
        if name =="monobit":
            return self ._cap_override ["monobit"]
        if name .startswith ("blockfreq"):
            return self ._cap_override ["blockfreq"]
        if name =="runs":
            return self ._cap_override ["runs"]
        if name .startswith ("cusum"):
            return self ._cap_override ["cusum"]
        if name .startswith ("longestrun"):
            return self ._cap_override ["longestrun"]
        if name =="fft":
            return self ._cap_override ["fft"]
        if name .startswith ("lincomp"):
            return self ._cap_override ["lincomp"]
        return int (self .BITS_PER_STREAM )

        # ---------------- RUN ----------------

    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        seed =int (getattr (self .config ,"seed",42 ))
        alpha =float (self .ALPHA )
        m =int (self .N_STREAMS )
        n_bits_req =int (self .BITS_PER_STREAM )
        key32 =self ._derive_key32 (seed )

        self .log (
        f"PF257 randomness tests: source={self.SOURCE}, streams={m}, bits/stream_req={n_bits_req}, "
        f"alpha={alpha}, bins={self.UNI_BINS}, longestrun={self.ENABLE_LONGESTRUN}, "
        f"fft={self.ENABLE_FFT}, lincomp={self.ENABLE_LINCOMP}"
        )

        tests :List [Tuple [str ,Callable [[np .ndarray ],float ]]]=[
        ("monobit",self ._p_monobit ),
        (f"blockfreq(M={self.BLOCK_FREQ_M})",lambda b :self ._p_block_freq (b ,M =self .BLOCK_FREQ_M )),
        ("runs",self ._p_runs ),
        ("cusum_fwd",lambda b :self ._p_cusum (b ,True )),
        ("cusum_bwd",lambda b :self ._p_cusum (b ,False )),
        ]
        if self .ENABLE_LONGESTRUN :
            tests .append (("longestrun(M=10000)",self ._p_longest_run ))
        if self .ENABLE_FFT :
            tests .append (("fft",self ._p_fft ))
        if self .ENABLE_LINCOMP :
            tests .append (("lincomp(BM,M=500)",lambda b :self ._linear_complexity_BM (b ,M =500 )))

        pvals :Dict [str ,List [float ]]={name :[]for name ,_ in tests }
        bits_used_by_test :Dict [str ,int ]={name :0 for name ,_ in tests }

        t0_total =time .perf_counter ()

        for i in range (m ):
            nonce =self ._derive_nonce63 (seed ,i )
            max_cap =max (self ._cap_for_test (name )for name ,_ in tests )
            n_bits_stream =min (n_bits_req ,int (max_cap ))

            bits_full =self ._keystream_bits (n_bits_stream ,key32 ,nonce )

            for name ,fn in tests :
                cap =int (self ._cap_for_test (name ))
                use =min (int (bits_full .size ),int (cap ))
                bits_used_by_test [name ]=use 
                p =float (fn (bits_full [:use ]))
                pvals [name ].append (p )

            if (i +1 )%max (1 ,m //5 )==0 :
                self .log (f"  progress: {i+1}/{m} streams")

        total_s =float (time .perf_counter ()-t0_total )

        rows =[]
        uni_detail_rows =[]

        for name ,_ in tests :
            arr =np .asarray (pvals [name ],dtype =np .float64 )
            used_bits =int (bits_used_by_test .get (name ,n_bits_req ))

            valid =np .isfinite (arr )
            arr_v =arr [valid ]
            m_valid =int (arr_v .size )

            # NOTE: (comment removed; repository is English-only)
            ci_low ,ci_high =self ._nist_proportion_ci (m_valid ,alpha )

            pass_rate =float (np .mean (arr_v >=alpha ))if m_valid >0 else float ("nan")
            pass_rate_pct =100.0 *pass_rate if np .isfinite (pass_rate )else float ("nan")
            prop_ok =bool (np .isfinite (pass_rate )and (pass_rate >=ci_low )and (pass_rate <=ci_high ))

            chi2_u ,p_u ,hist =self ._nist_pvalue_uniformity (arr_v ,bins =self .UNI_BINS )
            uni_ok =bool (np .isfinite (p_u )and (p_u >=self .P_UNI_THRESH ))
            nist_ok =bool ((m_valid >0 )and prop_ok and uni_ok )

            rows .append ({
            "Source":self .SOURCE ,
            "Test":name ,
            "Streams_total":m ,
            "Streams_valid":m_valid ,
            "Bits_per_stream_req":n_bits_req ,
            "Bits_used":used_bits ,
            "alpha":alpha ,

            "Pass_rate_%":pass_rate_pct ,
            "CI_low_%":100.0 *ci_low ,
            "CI_high_%":100.0 *ci_high ,
            "Proportion_OK":prop_ok ,

            "p_mean":float (np .nanmean (arr_v ))if m_valid >0 else float ("nan"),
            "p_min":float (np .nanmin (arr_v ))if m_valid >0 else float ("nan"),
            "p_5pct":float (np .nanpercentile (arr_v ,5 ))if m_valid >0 else float ("nan"),
            "p_95pct":float (np .nanpercentile (arr_v ,95 ))if m_valid >0 else float ("nan"),

            "Uni_bins":self .UNI_BINS ,
            "Uni_chi2":float (chi2_u ),
            "Uni_p":float (p_u ),
            "Uniformity_OK":uni_ok ,

            "NIST_OK":nist_ok ,
            "Time_total_s":total_s ,
            })

            uni_detail ={
            "Source":self .SOURCE ,
            "Test":name ,
            "Streams_valid":m_valid ,
            "Bits_used":used_bits ,
            "Uni_chi2":float (chi2_u ),
            "Uni_p":float (p_u ),
            }
            for k in range (self .UNI_BINS ):
                uni_detail [f"bin_{k}"]=int (hist [k ])
            uni_detail_rows .append (uni_detail )

            self ._save_uniformity_plot (f"{self.SOURCE}_{name}",hist )

        df =pd .DataFrame (rows )
        df_bins =pd .DataFrame (uni_detail_rows )

        self .save_results (df ,"randomness_pf257_nist_summary.xlsx")
        self .save_results (df_bins ,"randomness_pf257_nist_uniformity_bins.xlsx")

        # NOTE: (comment removed; repository is English-only)
        if hasattr (self ,"export_latex")and callable (getattr (self ,"export_latex")):
            self .export_latex (
            df .copy (),
            caption ="See README.md for usage and details.",
            label ="tab:pf257_nist_randomness",
            )

        self .log (f"Randomness test complete. Total time: {total_s:.2f} s")
        return df 