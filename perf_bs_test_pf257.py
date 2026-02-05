# perf_bs_test_pf257_v1.py
import time 
from dataclasses import dataclass 
from typing import List ,Sequence ,Tuple 

import numpy as np 
import matplotlib .pyplot as plt 

# NOTE: (comment removed; repository is English-only)
from pf257_egkem_spnctr_v1_x25519 import PF257_EGKEM_SPNCTR 


@dataclass 
class PerfRow :
    bs :int 
    ok :bool 
    enc_ms_mean :float 
    enc_ms_med :float 
    enc_ms_std :float 
    dec_ms_mean :float 
    dec_ms_med :float 
    dec_ms_std :float 
    err :str =""


def plot_bs_results (rows :Sequence [PerfRow ],save_path :str )->None :
    ok_rows =[r for r in rows if r .ok ]
    if not ok_rows :
        print("No successful measurements found; nothing to report.")
        return 

    bs_list =[r .bs for r in ok_rows ]
    enc_mean =[r .enc_ms_mean for r in ok_rows ]
    dec_mean =[r .dec_ms_mean for r in ok_rows ]

    x =np .arange (len (bs_list ))
    width =0.35 

    fig ,ax =plt .subplots (figsize =(10 ,6 ))
    rects1 =ax .bar (x -width /2 ,enc_mean ,width ,label ="Encryption (mean)")
    rects2 =ax .bar (x +width /2 ,dec_mean ,width ,label ="Deencryption (mean)")

    ax .set_xlabel ("Block Size (BS)")
    ax .set_ylabel ("Time (ms)")
    ax.set_title("PF257: timing vs block size")
    ax .set_xticks (x )
    ax .set_xticklabels ([str (b )for b in bs_list ])
    ax .legend ()
    ax .grid (axis ="y",linestyle ="--",alpha =0.5 )

    def autolabel (rects ):
        for r in rects :
            h =r .get_height ()
            ax .annotate (
            f"{h:.2f}",
            xy =(r .get_x ()+r .get_width ()/2 ,h ),
            xytext =(0 ,3 ),
            textcoords ="offset points",
            ha ="center",
            va ="bottom",
            fontsize =9 ,
            )

    autolabel (rects1 )
    autolabel (rects2 )

    fig .tight_layout ()
    plt .savefig (save_path ,dpi =300 )
    print (f"\nGrafik kaydedildi: {save_path}")
    plt .show ()


def analyze_bs_performance (
img_size :Tuple [int ,int ]=(256 ,256 ),
bs_list :Sequence [int ]=(3 ,4 ,16 ,32 ,64 ),
repeat_count :int =50 ,
warmup_count :int =5 ,
base_nonce :int =123456789 ,
use_unique_nonce :bool =True ,
auth :bool =True ,
verify :bool =True ,
save_plot_path :str ="performance_analysis_chart.png",
rng_seed :int =42 ,
)->List [PerfRow ]:

    rng =np .random .default_rng (rng_seed )
    original_img =rng .integers (0 ,256 ,size =img_size ,dtype =np .uint8 )

    rows :List [PerfRow ]=[]

    print("PF257 block-size performance benchmark (micro-benchmark)")
    print ("-"*98 )
    print (
    f"{'BS':<5} | {'Status':<8} | "
    f"{'Enc mean(ms)':>12} {'Enc med':>10} {'Enc std':>10} | "
    f"{'Dec mean(ms)':>12} {'Dec med':>10} {'Dec std':>10}"
    )
    print ("-"*98 )

    for bs in bs_list :
        try :
            cipher =PF257_EGKEM_SPNCTR (bs =int (bs ),auth =auth )

            # NOTE: (comment removed; repository is English-only)
            bob_priv ,bob_pub_raw =PF257_EGKEM_SPNCTR .kem_generate_keypair ()

            # --- Warm-up ---
            for w in range (warmup_count ):
                nonce =base_nonce +w if use_unique_nonce else base_nonce 
                packet ,_ =cipher .encrypt (original_img ,bob_pub_raw =bob_pub_raw ,nonce =nonce )
                dec_img =cipher .decrypt (
                packet ,
                bob_priv =bob_priv ,
                verify =verify ,
                bob_pub_raw =bob_pub_raw ,# NOTE: (comment removed; repository is English-only)
                )
                if not np .array_equal (dec_img ,original_img ):
                    raise ValueError("Decryption mismatch: lossless reconstruction failed.")

                    # NOTE: (comment removed; repository is English-only)
            enc_durs =np .empty (repeat_count ,dtype =np .float64 )
            dec_durs =np .empty (repeat_count ,dtype =np .float64 )

            last_dec =None 
            for i in range (repeat_count ):
                nonce =base_nonce +(warmup_count +i )if use_unique_nonce else base_nonce 

                t0 =time .perf_counter_ns ()
                packet ,_ =cipher .encrypt (original_img ,bob_pub_raw =bob_pub_raw ,nonce =nonce )
                t1 =time .perf_counter_ns ()
                enc_durs [i ]=(t1 -t0 )*1e-6 # ms

                t0 =time .perf_counter_ns ()
                last_dec =cipher .decrypt (
                packet ,
                bob_priv =bob_priv ,
                verify =verify ,
                bob_pub_raw =bob_pub_raw ,
                )
                t1 =time .perf_counter_ns ()
                dec_durs [i ]=(t1 -t0 )*1e-6 # ms

            if last_dec is None or (not np .array_equal (last_dec ,original_img )):
                raise ValueError("Decryption mismatch: lossless reconstruction failed.")

            row =PerfRow (
            bs =int (bs ),
            ok =True ,
            enc_ms_mean =float (enc_durs .mean ()),
            enc_ms_med =float (np .median (enc_durs )),
            enc_ms_std =float (enc_durs .std (ddof =1 ))if repeat_count >1 else 0.0 ,
            dec_ms_mean =float (dec_durs .mean ()),
            dec_ms_med =float (np .median (dec_durs )),
            dec_ms_std =float (dec_durs .std (ddof =1 ))if repeat_count >1 else 0.0 ,
            )
            rows .append (row )

            print (
            f"{row.bs:<5} | {'OK':<8} | "
            f"{row.enc_ms_mean:12.4f} {row.enc_ms_med:10.4f} {row.enc_ms_std:10.4f} | "
            f"{row.dec_ms_mean:12.4f} {row.dec_ms_med:10.4f} {row.dec_ms_std:10.4f}"
            )

        except Exception as e :
            row =PerfRow (
            bs =int (bs ),
            ok =False ,
            enc_ms_mean =0.0 ,enc_ms_med =0.0 ,enc_ms_std =0.0 ,
            dec_ms_mean =0.0 ,dec_ms_med =0.0 ,dec_ms_std =0.0 ,
            err =str (e ),
            )
            rows .append (row )
            print (f"{row.bs:<5} | {'FAIL':<8} | HATA: {row.err}")

    plot_bs_results (rows ,save_plot_path )
    return rows 


if __name__ =="__main__":
    analyze_bs_performance (
    img_size =(256 ,256 ),
    bs_list =(3 ,4 ,16 ,32 ,64 ),
    repeat_count =50 ,
    warmup_count =5 ,
    base_nonce =123456789 ,
    use_unique_nonce =True ,
    auth =True ,
    verify =True ,
    save_plot_path ="performance_analysis_chart.png",
    )
