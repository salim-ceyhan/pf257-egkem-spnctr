"""
pf257_experiment_base.py
Abstract base class for PF257 experiment modules.

Each experiment implements `run(images)` and writes its artifacts under:
    RESULTS_PF257_PAPER/exp_<id>_<name>/

This file is intentionally lightweight: it defines the common interface and
helper methods used by all experiment modules.
"""
from abc import ABC ,abstractmethod 
from pathlib import Path 
from typing import Dict ,Any 

import numpy as np 
import pandas as pd 

from pf257_experiment_framework import ExperimentConfig ,CipherWrapper ,LaTeXExporter 


class BaseExperiment (ABC ):
    """Abstract base class for PF257 experiment modules."""

    def __init__ (self ,config :ExperimentConfig ,cipher :CipherWrapper ,name :str ,description :str ):
        self .config =config 
        self .cipher =cipher 
        self .name =name 
        self .description =description 
        self .results :Dict [str ,Any ]={}
        self .output_dir =config .output_dir /f"exp_{name}"
        self .output_dir .mkdir (parents =True ,exist_ok =True )

    @abstractmethod 
    def run (self ,images :Dict [str ,np .ndarray ])->pd .DataFrame :
        ...

    def save_results (self ,df :pd .DataFrame ,filename :str )->Path :
        path =self .output_dir /filename 
        df .to_excel (path ,index =False ,engine ="xlsxwriter")
        print (f"✓ Results saved: {path}")
        return path 

    def export_latex (self ,df :pd .DataFrame ,caption :str ,label :str )->str :
        latex =LaTeXExporter .dataframe_to_latex (df ,caption ,label )
        LaTeXExporter .save_latex_table (latex ,self .output_dir /f"{self.name}_table.tex")
        return latex 

    def log (self ,msg :str )->None :
        print (f"[{self.name}] {msg}")
