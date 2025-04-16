import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple
import os
import json
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score
import numpy as np
import pandas as pd
from scipy import stats
import sys
from collections import defaultdict
import csv
import subprocess
from typing import Optional

def count_in_ranges(pi_values, range_size=0.1):
    """Count the number of `pi` values in 0.1 intervals from 0 to 1."""
    bins = np.arange(0, 1 + range_size, range_size)
    counts, _ = np.histogram(pi_values, bins=bins)
    return counts, bins

def prompt_func(data: str, prompt_type: str, query: str, text: str, desc: Optional[str] = None, narrative: Optional[str] = None) -> str:
    
    if data == 'robust04':
        if prompt_type == 'utility':
            return f"""
                Given a query and a web page, you must
                provide a score on an integer scale of 0 to 2 with the
                following meanings:
                2 = highly relevant, very helpful for this query
                1 = relevant, may be partly helpful but might contain
                other irrelevant content
                0 = not relevant, should never be shown for this query
                Assume that you are writing a report on the subject of the
                topic. If you would use any of the information contained
                in the web page in such a report, mark it 1. If the web page
                is primarily about the topic, or contains vital information
                about the topic, mark it 2. Otherwise, mark it 0.

                Query
                A person has typed "{query}" into a search engine.
                They were looking for: {desc} {narrative}

                Result
                Consider the following web page.
                —BEGIN WEB PAGE CONTENT—
                {text}
                —END WEB PAGE CONTENT—

                Instructions
                Produce a relevance score without providing any
                reasoning. (e.g., Relevance Score: {"{"}2{"}"})
                
                Result
                Relevance Score: {"{"}"""
        elif prompt_type == 'simple':
            return f"""Consider the following web page content:
                    -BEGIN WEB PAGE CONTENT-
                    {text}
                    -END WEB PAGE CONTENT-
                    Setting:
                    A person has typed "{query}" into a search engine.
                    This person's intent of this query was "{desc}"
                    Instruction:
                    Answer if the web content is relevant to the query. The seeker is {narrative}.
                    Answer {"{"}2{"}"} for highly relevant, {"{"}1{"}"} for somewhat or partially relevant, or {"{"}0{"}"} for not relevant.
                    Your answer: {"{"}"""

        elif prompt_type == 'rational':
            return f"""
                You are an expert judge of content. Using your internal knowledge
                and simple commonsense reasoning, try to verify if the passage is
                relevant to the query.
                Here, "0" represents that the passage has nothing to do with the
                query, "1" represents that the passage somewhat or partially
                answers the query, and "2" represents that the
                passage is dedicated to the query and contains an answer.
                
                Give your answer from one of the categories 0, 1 or 2 only. One of the
                categorical values is compulsory in the answer.
                Instructions: Think about the question. Provide your answer in terms of 0, 1, or 2
                categories. Produce a relevance score without providing any
                reasoning. (e.g., Relevance Score: {"{"}2{"}"})
                ###
                Query: {query}
                They were looking for: {desc} {narrative}
                Passage: {text}
                
                Result
                Relevance Score: {"{"}"""
            
    elif data == 'gcovid5':
        if prompt_type == 'utility':
            return f"""
                Given a query and a web page, you must
                provide a score on an integer scale of 0 to 2 with the
                following meanings:
                2 = highly relevant, very helpful for this query
                1 = relevant, may be partly helpful but might contain
                other irrelevant content
                0 = not relevant, should never be shown for this query
                Assume that you are writing a report on the subject of the
                topic. If you would use any of the information contained
                in the web page in such a report, mark it 1. If the web page
                is primarily about the topic, or contains vital information
                about the topic, mark it 2. Otherwise, mark it 0.

                Query
                A person has typed "{query}" into a search engine.
                The user wanted to know "{desc}"
                The user is {narrative}.

                Result
                Consider the following web page.
                —BEGIN WEB PAGE CONTENT—
                {text}
                —END WEB PAGE CONTENT—

                Instructions
                Produce a relevance score without providing any
                reasoning. (e.g., Relevance Score: {"{"}2{"}"})
                
                Result
                Relevance Score: {"{"}"""
        elif prompt_type == 'simple':
            return f"""Consider the following web page content:
                    -BEGIN WEB PAGE CONTENT-
                    {text}
                    -END WEB PAGE CONTENT-
                    Setting:
                    A person has typed "{query}" into a search engine.
                    This person's intent of this query was "{desc}"
                    Instruction:
                    Answer if the web content is relevant to the query. The seeker is {narrative}.
                    Answer {"{"}2{"}"} for highly relevant, {"{"}1{"}"} for somewhat or partially relevant, or {"{"}0{"}"} for not relevant.
                    Your answer: {"{"}"""

        elif prompt_type == 'rational':
            return f"""
                You are an expert judge of content. Using your internal knowledge
                and simple commonsense reasoning, try to verify if the passage is
                relevant to the query.
                Here, "0" represents that the passage has nothing to do with the
                query, "1" represents that the passage somewhat or partially
                answers the query, and "2" represents that the
                passage is dedicated to the query and contains an answer.
                
                Give your answer from one of the categories 0, 1 or 2 only. One of the
                categorical values is compulsory in the answer.
                Instructions: Think about the question. Provide your answer in terms of 0, 1, or 2
                categories. Produce a relevance score without providing any
                reasoning. (e.g., Relevance Score: {"{"}2{"}"})
                ###
                Query: {query}
                The user wanted to know "{desc}"
                The user is {narrative}.
                Passage: {text}
                
                Result
                Relevance Score: {"{"}"""
                
    elif data == 'adhoc7' or data=='adhoc8':
        if prompt_type == 'utility':
            return f"""
                Given a query and a web page, you must
                provide a Yes or No answer of whether the query is relevant to the web page.
                Assume that you are writing a report on the subject of the
                topic. If the webpage contains some vital information
                about the topic, answer Yes. Otherwise, mark it No.

                Query
                A person has typed "{query}" into a search engine.
                They were looking for: {desc} {narrative}

                Result
                Consider the following web page.
                —BEGIN WEB PAGE CONTENT—
                {text}
                —END WEB PAGE CONTENT—

                Instructions
                Is the web content relevant to the query? (Answer yes or no.)
                Answer yes or no.
                Your answer:"""
        elif prompt_type == 'simple':
            return f"""Consider the following web page content:
                -BEGIN WEB PAGE CONTENT-
                {text}
                -END WEB PAGE CONTENT-
                Setting:
                A person has typed "{query}" into a search engine.
                This person's intent of this query was "{desc}"
                Instruction:
                Answer if the web content is relevant to the query. {narrative}
                Answer yes or no.
                Your answer:"""
        elif prompt_type == 'rational':
            return f"""
                You are an expert judge of content. Using your internal knowledge
                and simple commonsense reasoning, try to verify if the passage is
                relevant to the query.
                Here, "No" represents that the passage has nothing to do with the
                query, and "Yes" represents that the
                passage is dedicated to the query and contains an answer.
                
                Give your answer from one of the categories Yes or No only.
                Instructions: Think about the question. Provide your answer in terms of Yes or No.
                categories. Answer if the passage is relevant to the query without providing any
                reasoning.
                ###
                Query: {query}
                They were looking for: {desc} {narrative}
                Passage: {text}
                
                Instruction
                Is the web content relevant to the query? (Answer yes or no.)
                Your answer:"""

def get_runs(track:str) -> List[str]:
    if track == 'adhoc8':
        return ["1", "att99atde", "Flab8at", "ibmg99c", "isa50t", "MITSLStdn", "plt8ah1", "Sab8A4",
            "UniNET8St", "8manexT3D1N0", "att99ate", "Flab8atd2", "ibms99a", "kdd8ps16",
            "nttd8al", "plt8ah2", "Scai8Adhoc", "UT800", "acsys8aln2", "cirtrc82", "Flab8atdn",
            "ibms99b", "kdd8qe01", "nttd8ale", "plt8ah3", "surfahi1", "UT803", "acsys8alo",
            "CL99SD", "Flab8ax", "ibms99c", "kdd8sh16", "nttd8alx", "plt8ah4", "surfahi2",
            "UT803b", "acsys8alo2", "CL99SDopt1", "ic99dafb", "kuadhoc", "nttd8am", "plt8ah5",
            "surffal2", "UT810", "acsys8amn", "CL99SDopt2", "fub99a", "iit99au1", "mds08a1",
            "nttd8ame", "READWARE", "tno8d3", "UT813", "acsys8asn", "CL99XT", "fub99td", "iit99au2",
            "mds08a2", "ok8alx", "READWARE2", "tno8d4", "uwmt8a0", "AntHoc1", "CL99XTopt", "fub99tf",
            "iit99ma1", "mds08a3", "ok8amxc", "ric8dnx", "tno8t2", "uwmt8a1", "apl8c221",
            "disco1", "fub99tt", "INQ601", "mds08a4", "ok8asxc", "ric8dpn", "UB99SW", "uwmt8a2",
            "apl8c621", "Dm8Nbn", "GE8ATD3", "INQ602", "mds08a5", "orcl99man", "ric8dpx",
            "UB99T", "weaver1", "apl8ctd", "Dm8NbnR", "GE8ATDN1", "INQ603", "Mer8Adtd1", "pir9Aa1",
            "ric8tpn", "umd99a1", "weaver2", "apl8n", "Dm8TFbn", "GE8ATDN2", "INQ604", "Mer8Adtd2",
            "pir9Aatd", "ric8tpx", "unc8al32", "apl8p", "Dm8TFidf", "GE8MTD2", "isa25", "Mer8Adtd4",
            "pir9At0", "Sab8A1", "unc8al42", "att99atc", "ibmg99a", "isa25t", "Mer8Adtnd3",
            "pir9Atd0", "Sab8A2", "unc8al52", "att99atdc", "Flab8as", "ibmg99b", "isa50",
            "MITSLStd", "pir9Attd", "Sab8A3", "UniNET8Lg"]
    elif track == 'dl2020':
        return ["BIT-run1", "BIT-run2", "BIT-run3", "ICIP_run1", "ICIP_run2", "ICIP_run3",
            "RMIT_DFRee", "RMIT_DPH", "TUW-TKL-2k", "TUW-TKL-4k", "bcai_bertb_docv",
            "bcai_classic", "bigIR-DH-T5-F", "bigIR-DH-T5-R", "bigIR-DT-T5-F",
            "bigIR-DT-T5-R", "bigIR-DTH-T5-F", "bigIR-DTH-T5-R", "bl_bcai_model1",
            "bl_bcai_multfld", "bl_bcai_prox", "d_bm25", "d_bm25rm3", "d_d2q_bm25",
            "d_d2q_bm25rm3", "d_d2q_duo", "d_rm3_duo", "fr_doc_roberta", "indri-sdmf",
            "longformer_1", "mpii_run1", "mpii_run2", "mpii_run3", "ndrm1-full",
            "ndrm1-re", "ndrm3-full", "ndrm3-orc-full", "ndrm3-orc-re", "ndrm3-re",
            "nlm-bm25-prf-1", "nlm-bm25-prf-2", "rindri-bm25", "rmit_indri-fdm",
            "rmit_indri-sdm", "roberta-large", "rterrier-dph", "rterrier-dph_sd",
            "rterrier-expC2", "rterrier-tfidf", "rterrier-tfidf2", "terrier-jskls",
            "uob_runid1", "uob_runid2", "uob_runid3", "uogTr31oR", "uogTrBaseDPH",
            "uogTrBaseDPHQ", "uogTrBaseL16", "uogTrBaseL17o", "uogTrBaseQL16",
            "uogTrBaseQL17o", "uogTrQCBMP", "uogTrT20"]
    elif track=='covid5' or track=='gcovid5':
        return ["BRPHJ_BM25", "BRPHJ_bert", "BRPHJ_logistic", "BioInfo-run1", "BioInfo-run2",
            "BioInfo-run3", "BioInfo-run4", "BioInfo-run5", "BioInfo-run6", "CSIROmedFR",
            "CSIROmedNIP", "CSIROmedNIR", "CincyMedIR-0-2-3-4", "CincyMedIR-0-4-1-3",
            "CincyMedIR-1", "CincyMedIR-1-2-1-3", "CincyMedIR-1-4-1-3", "CincyMedIR-1-6-4-3",
            "CincyMedIR-20-5-4", "CincyMedIR-s-20-5-4", "DoRAWithJudgments_1k",
            "DoRAWithJudgments_6k", "DoRA_MSMARCO_1k", "DoRA_MSMARCO_1k_C", "DoRA_MSMARCO_6k",
            "DoRA_NO_Judgments_1k", "DoRA_NO_Judgments_6k", "HKPU-BM25-dPRF", "HKPU-Gos1-dPRF",
            "HKPU-LGD-dPRF", "HKPU-MATF-dPRF", "HKPU-MVD-dPRF", "HKPU-PL2-dPRF", "HKPU-RM3-dPRF",
            "HKPU-SPUD-dPRF", "MacEwan-base", "SFDC-enc45-refus12", "UIowaS_Run1", "UIowaS_Run2",
            "UIowaS_Run3", "UPrrf102-r5", "UPrrf102-wt-r5", "UPrrf80-r5", "UPrrf89-r5",
            "UPrrf93-r5", "UPrrf93-wt-r5", "bm25L1_bilstm_run", "bm25L1_linear_run",
            "bm25L2_bilstm_run", "bm25L2_linear_run", "bm25L_bilstm_run", "bm25L_bl_run5",
            "bm25_bl_run5", "covidex.r5.1s", "covidex.r5.1s.lr", "covidex.r5.2s",
            "covidex.r5.2s.lr", "covidex.r5.d2q.1s", "covidex.r5.d2q.1s.lr", "covidex.r5.d2q.2s",
            "covidex.r5.d2q.2s.lr", "elhuyar_prf_nof99d", "elhuyar_prf_nof9p",
            "elhuyar_prf_nof99p", "elhuyar_rrf_nof09p", "elhuyar_rrf_nof99p", "fc3-qrel-hidden",
            "final.qruir.f.txt", "final.qruir.txt", "final.qruir33.txt", "jlbasernd5-jlQErnd5",
            "mpiid5_run1", "mpiid5_run2", "poznan_baseline", "poznan_rerank1", "poznan_rerank2",
            "r5.d2q.fusion1", "r5.d2q.fusion2", "r5.d2q.qqabs", "r5.d2q.rf", "r5.fusion1",
            "r5.fusion2", "r5.qqabs", "r5.rf", "rk_bdl_brx_logit", "rk_bm25_bs",
            "rk_bm25_dfr_lmd_rrf", "rk_ir_bdl_trf_brx_lm", "rk_ir_bdl_trf_brx_rr",
            "rk_ir_trf_logit_rr", "rk_trf_brx_rrf", "run1_C_Arf_SciB", "run2_C-Arf_SciB",
            "sab20.5.1.meta.docs", "sab20.5.2.dfo.metado", "sab20.5.2.meta.docs", "sab20.5.3.dfo",
            "sab20.5.3.metadocs_m", "sab20.5.4.dfo", "sab20.5.dfo", "sab20.5.metadocs_m",
            "uab.base", "uab.idf", "ucd_cs_r1", "ucd_cs_r2", "ucd_cs_r3", "udel_fang_ltr_split",
            "udel_fang_ltr_uni", "udel_fang_nir", "uogTrDPH_QE_RF", "uogTrDPH_QE_RF_CB",
            "uogTrDPH_QE_RF_SB", "uogTrDPH_QE_RF_SB_B", "uogTrDPH_QE_RF_SB_CB", "uogTrDPH_QE_SB",
            "uogTrDPH_QE_SB_B", "uogTrDPH_QE_SB_CB", "uw_base1", "uw_base2", "uw_crowd1",
            "uw_crowd2", "uw_fb1", "uw_fb2", "xj4wang_run1", "xj4wang_run2", "xj4wang_run3"]
    elif track=='robust04' or track=='robust04-2':
        return [
            "JuruDes", "JuruDesAggr", "JuruDesLaMd", "JuruDesQE", "JuruDesSwQE", "JuruDesTrSl", 
            "JuruTit", "JuruTitDes", "JuruTitSwDs", "JuruTitSwQE", "NLPR04COMB", "NLPR04LMts", 
            "NLPR04LcA", "NLPR04NcA", "NLPR04OKapi", "NLPR04SemLM", "NLPR04clus10", "NLPR04clus9", 
            "NLPR04okall", "NLPR04okdiv", "NLPR04oktwo", "SABIR04BA", "SABIR04BD", "SABIR04BT", 
            "SABIR04FA", "SABIR04FD", "SABIR04FT", "apl04rsDw", "apl04rsTDNfw", "apl04rsTDNw5", 
            "apl04rsTs", "apl04rsTw", "fub04De", "fub04Dg", "fub04Dge", "fub04T2ge", "fub04TDNe", 
            "fub04TDNg", "fub04TDNge", "fub04Te", "fub04Tg", "fub04Tge", "humR04d4", "humR04d4e5", 
            "humR04d5", "humR04d5i", "humR04d5m", "humR04t1", "humR04t1i", "humR04t1m", "humR04t5", 
            "humR04t5e1", "icl04pos2d", "icl04pos2f", "icl04pos2t", "icl04pos2td", "icl04pos48f", 
            "icl04pos7f", "icl04pos7tap", "icl04pos7tat", "icl04pos7td", "mpi04r01", "mpi04r02", 
            "mpi04r04", "mpi04r05", "mpi04r06", "mpi04r07", "mpi04r08", "mpi04r09", "mpi04r10", 
            "mpi04r11", "pircRB04d2", "pircRB04d3", "pircRB04d4", "pircRB04d5", "pircRB04t1", 
            "pircRB04t2", "pircRB04t3", "pircRB04t4", "pircRB04td2", "pircRB04td3", "polyudp2", 
            "polyudp4", "polyudp5", "polyudp6", "polyutp1", "polyutp3", "uic0401", "uogRobDBase", 
            "uogRobDWR10", "uogRobDWR5", "uogRobLBase", "uogRobLT", "uogRobLWR10", "uogRobLWR5", 
            "uogRobSBase", "uogRobSWR10", "uogRobSWR5", "vtumdesc", "vtumlong252", "vtumlong254", 
            "vtumlong344", "vtumlong348", "vtumlong432", "vtumlong436", "vtumtitle", "wdo25qla1", 
            "wdoqdn1", "wdoqla1", "wdoqsn1"
        ]
    elif track=='adhoc7':
        return [
            "acsys7al", "acsys7as", "acsys7mi", "AntHoc01", "APL985L",
            "APL985LC", "APL985SC", "att98atc", "att98atdc", "att98atde",
            "bbn1", "Brkly24", "Brkly25", "Brkly26", "CLARIT98CLUS",
            "CLARIT98COMB", "CLARIT98RANK", "Cor7A1clt", "Cor7A2rrd", "Cor7A3rrf",
            "dsir07a01", "dsir07a02", "ETHAB0", "ETHAC0", "ETHAR0",
            "FLab7ad", "FLab7at", "FLab7atE", "fsclt7a", "fsclt7m",
            "fub98a", "fub98b", "gersh1", "gersh2", "gersh3",
            "harris1", "ibmg98a", "ibmg98b", "ibmg98c", "ibms98a",
            "ibms98b", "ibms98c", "ic98san3", "ic98san4", "iit98au1",
            "iit98au2", "iit98ma1", "INQ501", "INQ502", "INQ503", 
            "iowacuhk1", "iowacuhk2", "jalbse011", "jalbse012", "jalbse013",
            "KD70000", "KD71010q", "KD71010s", "kslsV1", "lanl981",
            "LIAClass", "LIArel2", "LIAshort2", "LNaTit7", "LNaTitDesc7",
            "LNmanual7", "mds98t", "mds98t2", "mds98td", "MerAdRbtd",
            "MerAdRbtnd", "MerTetAdtnd", "nectitech", "nectitechall", "nectitechdes",
            "nsasgrp3", "nsasgrp4", "nthu1", "nthu2", "nthu3",
            "nttdata7Al0", "nttdata7Al2", "nttdata7At1", "ok7am", "ok7as",
            "ok7ax", "pirc8Aa2", "pirc8Ad", "pirc8At", "ScaiTrec7",
            "t7miti1", "tno7cbm25", "tno7exp1", "tno7tw4", "umd98a1",
            "umd98a2", "unc7aal1", "unc7aal2", "uoftimgr", "uoftimgu",
            "uwmt7a0", "uwmt7a1", "uwmt7a2"
        ]
    elif track=='covid4':
        return [
            "BITEM_BERT4", "BITEM_COVOC4", "BioInfo-run1", "BioInfo-run2", "BioInfo-run3",
            "CSIROmedBM", "CSIROmedNIR", "CSIROmedNO", "CincyMedIR-7", "CincyMedIR-8",
            "CincyMedIR-9", "Emory_rnd4_run1", "Emory_rnd4_run2", "Emory_rnd4_run3", "HKPU-Gos1-pPRF",
            "HKPU-MATF-pPRF", "HKPU-SPUD-pPRF", "ILPS_UvA_allrounds_c", "ILPS_UvA_big_diverse", "ILPS_UvA_zeroshot_BE",
            "Marouane_QQ_EnM-BTN", "Marouane_eQQ_EnM-BTN", "OHSU_R4_totalfusion", "OHSU_TF_UDGEN_AVG", "OHSU_totalfusion_avg",
            "SFDC-fus12-enc34", "SFDC-re-fus12-enc24", "SFDC-re-fus12-enc34", "UPrrf38rrf2-r4", "UPrrf38rrf3-r4",
            "UPrrf38rrf3v2-r4", "active_learning", "aserini2000-t53", "bm25_bertsim_run4", "bm25_bl_run4",
            "combined", "covidex.r4.d2q.duot5", "covidex.r4.duot5", "covidex.r4.duot5.lr", "jlQErnd4",
            "jlbasernd4", "jlbasernd4-jlQErnd4", "l2r", "mpiid5_run1", "poznan_p4_run1",
            "poznan_p4_run2", "poznan_p4_run3", "r4.fusion1", "r4.fusion2", "r4.rf",
            "sab20.4.dfo", "sab20.4.metadocs_m", "sab20.4.rocchio", "uab.run1", "ucd_cs_r4_r1",
            "ucd_cs_r4_r2", "ucd_cs_r4_r3", "udel_fang_lambdarank", "udel_fang_ltr_nobert", "udel_fang_nir",
            "uogTrDPH_QE", "uogTrDPH_QE_SCB1", "uogTrDPH_QE_SCB_PM1", "uw_base", "uw_crowd",
            "uw_fb", "xj4wang_run1", "xj4wang_run2", "xj4wang_run3"
        ]

def convert_df_to_qrel(input_df, output_qrel):
    with open(output_qrel, 'w') as qrel_file:
        for _, row in input_df.iterrows():
            topic_id = row['topic_id']
            doc_id = row['doc_id']
            relevance = row['pi']
            # make relevance int
            relevance = int(relevance)
            qrel_file.write(f"{topic_id} 0 {doc_id} {relevance}\n")

def convert_df_to_qrel_true(input_df, output_qrel):
    with open(output_qrel, 'w') as qrel_file:
        for _, row in input_df.iterrows():
            topic_id = row['topic_id']
            doc_id = row['doc_id']
            relevance = row['annotation']
            # make relevance int
            qrel_file.write(f"{topic_id} 0 {doc_id} {relevance}\n")

def read_trec_eval_file(file_path, metrics):
    data = {'all': {}, 'topics': {}}
    with open(file_path, 'r') as f:
        for line in f:
            metric, topic, value = line.strip().split()
            if metric in metrics:
                value = float(value.strip("'"))
                if topic == 'all':
                    data['all'][metric] = value
                else:
                    data['topics'].setdefault(topic, {})[metric] = value
    return data
    
def compute_correlations(true_scores, model_scores):
    # Check if lengths are different
    if len(true_scores) != len(model_scores):
        print(f"WARNING: Score length mismatch detected! true_scores: {len(true_scores)}, model_scores: {len(model_scores)}")
        # Take the shorter length
        min_len = min(len(true_scores), len(model_scores))
        true_scores = true_scores[:min_len]
        model_scores = model_scores[:min_len]
        print(f"Truncated both scores to length {min_len} for computation")
    
    tau, _ = stats.kendalltau(true_scores, model_scores)
    pearson, _ = stats.pearsonr(true_scores, model_scores)
    return tau, pearson

def compute_max_rank_drop(true_scores, model_scores, system_names):
    """
    A more concise Pandas-based approach to compute the maximum drop in rank.

    Parameters
    ----------
    true_scores : List[float]
        Scores of systems under 'true' qrels.
    model_scores : List[float]
        Scores of systems under 'model' qrels.
    system_names : List[str]
        Identifiers (tags) for each system. Must have same length as the scores.

    Returns
    -------
    float
        The maximum (worst) rank drop across all systems.
    """
    # Build a DataFrame with columns: system, true_score, model_score
    df = pd.DataFrame({
        'system': system_names,
        'true_score': true_scores,
        'model_score': model_scores
    })

    # Rank (descending) based on true_score. For ties, you can choose your
    # preferred method: 'first', 'min', 'dense', etc. 
    # Here, we use 'min', so all ties get the same rank.
    df['rank_true'] = df['true_score'].rank(ascending=False, method='min')

    # Rank (descending) based on model_score
    df['rank_model'] = df['model_score'].rank(ascending=False, method='min')

    # Calculate the difference
    df['rank_drop'] = df['rank_model'] - df['rank_true']

    # Return the maximum drop
    return df['rank_drop'].max() if len(df) > 0 else 0.0

def compute_max_rank_drop_top5(true_scores, model_scores, system_names):
    """
    Compute the maximum rank drop, but only among the top 5 systems
    in the 'true' ranking.

    Parameters
    ----------
    true_scores : List[float]
        A list of the 'true' scores for each system (same order as system_names).
    model_scores : List[float]
        A list of the 'model' scores for each system (same order as system_names).
    system_names : List[str]
        The list of system identifiers or names.

    Returns
    -------
    float
        The maximum (worst) rank drop within the top 5 systems of the true ranking.
    """
    # 1. Build a DataFrame of systems and their scores
    df = pd.DataFrame({
        'system': system_names,
        'true_score': true_scores,
        'model_score': model_scores
    })

    # 2. Rank by true_score (descending)
    df['rank_true'] = df['true_score'].rank(ascending=False, method='min')
    # 3. Rank by model_score (descending)
    df['rank_model'] = df['model_score'].rank(ascending=False, method='min')

    # 4. Keep only the top 5 systems in true ranking
    df_top5 = df[df['rank_true'] <= 5].copy()

    # 5. If we have fewer than 5 systems or none, handle gracefully
    if df_top5.empty:
        return 0.0

    # 6. Calculate rank drop (model - true)
    df_top5['rank_drop'] = df_top5['rank_model'] - df_top5['rank_true']

    # 7. Return the maximum drop
    return df_top5['rank_drop'].max()

def analyze_runs(true_dir, model_dir, metrics, model):
    def load_data(directory):
        return {
            file.replace('-eval-True', '').replace(f'-eval-{model}', ''):
                read_trec_eval_file(os.path.join(directory, file), metrics)
            for file in os.listdir(directory)
            if file.endswith(('-eval-True', f'-eval-{model}'))
        }

    true_data = load_data(true_dir)
    model_data = load_data(model_dir)
    common_runs = set(true_data.keys()) & set(model_data.keys())
    results = {}

    for metric in metrics:
        # Collect topic-level scores
        true_topic_scores = []
        model_topic_scores = []
        topic_system_names = []

        # Collect system-level scores
        true_all_scores = []
        model_all_scores = []
        all_system_names = []

        # Gather scores
        for run in common_runs:
            # Topic-level
            common_topics = (set(true_data[run]['topics']) 
                             & set(model_data[run]['topics']))
            if common_topics:
                t_avg = np.mean([true_data[run]['topics'][t][metric] 
                                 for t in common_topics])
                m_avg = np.mean([model_data[run]['topics'][t][metric] 
                                 for t in common_topics])
                true_topic_scores.append(t_avg)
                model_topic_scores.append(m_avg)
                topic_system_names.append(run)

            # System-level ('all')
            if (metric in true_data[run]['all'] 
                and metric in model_data[run]['all']):
                true_all_scores.append(true_data[run]['all'][metric])
                model_all_scores.append(model_data[run]['all'][metric])
                all_system_names.append(run)

        # Compute correlations, p@10, and rank drops
        try:
            # --- Topics ---
            if true_topic_scores and model_topic_scores:
                tau, pearson = compute_correlations(true_topic_scores, 
                                                    model_topic_scores)
                p_at_10 = compute_precision_at_k(true_topic_scores, 
                                                 model_topic_scores)
                drop_topic = compute_max_rank_drop(true_topic_scores, 
                                                   model_topic_scores, 
                                                   topic_system_names)
                drop5_topic = compute_max_rank_drop_top5(true_topic_scores, 
                                                        model_topic_scores, 
                                                        topic_system_names)

                results[f'tau_{metric}_topic'] = tau
                results[f'pearson_{metric}_topic'] = pearson
                results[f'p@10_{metric}_topic'] = p_at_10
                results[f'drop_{metric}_topic'] = drop_topic
                results[f'drop5_{metric}_topic'] = drop5_topic
            else:
                # If no overlap in topics, set default to 0.0
                results[f'tau_{metric}_topic'] = 0.0
                results[f'pearson_{metric}_topic'] = 0.0
                results[f'p@10_{metric}_topic'] = 0.0
                results[f'drop_{metric}_topic'] = 0.0
                results[f'drop5_{metric}_topic'] = 0.0

            # --- System-level ---
            if true_all_scores and model_all_scores:
                tau, pearson = compute_correlations(true_all_scores, 
                                                    model_all_scores)
                p_at_10 = compute_precision_at_k(true_all_scores, 
                                                 model_all_scores)
                drop_system = compute_max_rank_drop(true_all_scores, 
                                                    model_all_scores, 
                                                    all_system_names)
                drop5_system = compute_max_rank_drop_top5(true_all_scores,
                                                        model_all_scores,
                                                        all_system_names)
                

                results[f'tau_{metric}_system'] = tau
                results[f'pearson_{metric}_system'] = pearson
                results[f'p@10_{metric}_system'] = p_at_10
                results[f'drop_{metric}_system'] = drop_system
                results[f'drop5_{metric}_system'] = drop5_system
            else:
                results[f'tau_{metric}_system'] = 0.0
                results[f'pearson_{metric}_system'] = 0.0
                results[f'p@10_{metric}_system'] = 0.0
                results[f'drop_{metric}_system'] = 0.0
                results[f'drop5_{metric}_system'] = 0.0

        except Exception as e:
            print(f"WARNING: Error processing metric {metric}: {str(e)}")
            # Safely set defaults
            for suffix in ["topic", "system"]:
                results[f'tau_{metric}_{suffix}'] = 0.0
                results[f'pearson_{metric}_{suffix}'] = 0.0
                results[f'p@10_{metric}_{suffix}'] = 0.0
                results[f'drop_{metric}_{suffix}'] = 0.0
                results[f'drop5_{metric}_{suffix}'] = 0.0

    return results


def compute_precision_at_k(true_scores, model_scores, k=10):
    true_ranks = np.argsort(true_scores)[::-1][:k]
    model_ranks = np.argsort(model_scores)[::-1][:k]
    return len(set(true_ranks) & set(model_ranks)) / k

def calculate_reljudge_metrics(metrics: List[str], model: str, root:str, df: pd.DataFrame, track: str) -> Dict[str, float]:
    basic_metrics = calculate_basic_metrics(df)
    treceval = "../trec_eval-main/trec_eval" # download and makefile the trec_eval framework here!!
    convert_df_to_qrel(df, f"{root}/{model}qrels.{track}")
    qrels = f"{root}/{model}qrels.{track}"
    resultsdir = f"../data/{track}/runs"
    runs = get_runs(track=track)
    evaldir = f"{root}/{model}"
    if not os.path.exists(evaldir):
        os.makedirs(evaldir)
    for tag in runs:
        input_file = f"{resultsdir}/input.{tag}"
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Can't find input file for {tag}")

        outfile = f"{evaldir}/{tag}-eval-{model}"
        command = [treceval, "-m", "all_trec", "-c", "-q", qrels, input_file]

        with open(outfile, 'w') as out:
            try:
                subprocess.run(command, check=True, stdout=out, stderr=subprocess.PIPE)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"trec_eval failed for {tag}: {e.stderr.decode()}")
    
    true_dir = f"{root}/True"
    if not os.path.exists(true_dir):
        convert_df_to_qrel_true(df, f"{root}/qrels.true")
        true_qrels = f"{root}/qrels.true"
        if not os.path.exists(true_dir):
            os.makedirs(true_dir)
        for tag in runs:
            input_file = f"{resultsdir}/input.{tag}"
            if not os.path.exists(input_file):
                raise FileNotFoundError(f"Can't find input file for {tag}")

            outfile = f"{true_dir}/{tag}-eval-True"
            command = [treceval, "-m", "all_trec", "-c", "-q", true_qrels, input_file]

            with open(outfile, 'w') as out:
                try:
                    subprocess.run(command, check=True, stdout=out, stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"trec_eval failed for {tag}: {e.stderr.decode()}")

    model_dir = f"{root}/{model}"
    
    results = analyze_runs(true_dir, model_dir, metrics, model)
    
    all_metrics = {**basic_metrics, **results}
    
    return all_metrics

def save_intermediate_results(b_list: List[int], all_metric_scores: List[List[Dict[str, float]]], labels: List[str], save_dir: str, iteration: int):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all unique metric names
    all_metrics = set()
    for experiment_metrics in all_metric_scores:
        for metric_dict in experiment_metrics:
            all_metrics.update(metric_dict.keys())
    
    # Plot each metric
    for metric in all_metrics:
        metric_scores = [[score[metric] for score in experiment_metrics] for experiment_metrics in all_metric_scores]
        plot_path = os.path.join(save_dir, f'{metric}_plot_intermediate_{iteration}.png')
        plot_metric(b_list, metric_scores, labels, metric, plot_path)
    
    # Save intermediate results to JSON
    results_dict = {
        label: {
            str(B): {metric: scores[i][metric] for metric in scores[i]}
            for i, B in enumerate(b_list)
        }
        for label, scores in zip(labels, all_metric_scores)
    }
    
    json_path = os.path.join(save_dir, f'intermediate_metrics_{iteration}.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Intermediate results for iteration {iteration} saved in {save_dir}")


def calculate_pyramid_metrics(df):
    basic_metrics = calculate_basic_metrics(df)
    human_scores = calculate_pyramid_score(df, 'annotation')
    llm_scores = calculate_pyramid_score(df, 'pi')
    topic_metrics = []
    for topic in human_scores.keys():
        topic_metrics.append(calculate_ranking_metrics(human_scores[topic], llm_scores[topic]))
    avg_topic_metrics = np.mean(topic_metrics, axis=0)
    overall_human_scores = {system: np.mean([scores[system] for scores in human_scores.values() if system in scores])
                            for system in set().union(*human_scores.values())}
    overall_llm_scores = {system: np.mean([scores[system] for scores in llm_scores.values() if system in scores])
                          for system in set().union(*llm_scores.values())}
    overall_metrics = calculate_ranking_metrics(overall_human_scores, overall_llm_scores)
    # make all metrics into one dictionary
    all_metrics = {**basic_metrics, **{'AP@5': avg_topic_metrics[0], 'AP@10': avg_topic_metrics[1],
                                       'Kendall\'s Tau': avg_topic_metrics[2], 'Pearson\'s R': avg_topic_metrics[3],
                                       'Spearman\'s R': avg_topic_metrics[4], 'Overall AP@5': overall_metrics[0],
                                       'Overall AP@10': overall_metrics[1], 'Overall Kendall\'s Tau': overall_metrics[2],
                                       'Overall Pearson\'s R': overall_metrics[3], 'Overall Spearman\'s R': overall_metrics[4]}}
    return all_metrics

def numpy_safe_json(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_safe_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_safe_json(item) for item in obj]
    else:
        return obj

def save_results(b_list: List[int], all_metric_scores: List[List[Dict[str, Any]]], labels: List[str], save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    
    # Get all unique metric names
    all_metrics = set()
    for experiment_metrics in all_metric_scores:
        for metric_dict in experiment_metrics:
            all_metrics.update(metric_dict.keys())
    
    # Plot each metric
    for metric in all_metrics:
        metric_scores = [[score.get(metric, float('nan')) for score in experiment_metrics] for experiment_metrics in all_metric_scores]
        plot_path = os.path.join(save_dir, f'{metric}.png')
        plot_metric(b_list, metric_scores, labels, metric, plot_path)
    
    # Save all results to JSON
    results_dict = {
        label: {
            str(B): numpy_safe_json(scores[i])
            for i, B in enumerate(b_list)
        }
        for label, scores in zip(labels, all_metric_scores)
    }
    
    json_path = os.path.join(save_dir, 'all_metrics.json')
    with open(json_path, 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"Results saved in {save_dir}")

def plot_metric(b_list: List[int], metric_scores: List[List[float]], labels: List[str], metric_name: str, save_path: str):
    plt.figure(figsize=(12, 8))
    b_list_str = [str(b) for b in b_list]
    
    for scores, label in zip(metric_scores, labels):
        plt.plot(b_list_str, scores, marker='o', label=label, linewidth=2, markersize=6)
    
    plt.xlabel('Budget (B)', fontsize=14)
    plt.ylabel(metric_name.capitalize(), fontsize=14)
    plt.title(f"{metric_name.capitalize()} vs. Budget (B)", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
def calculate_pyramid_score(df, column):
    scores = {}
    for name, group in df.groupby(['folder_name', 'file_name']):
        topic, file_name = name
        system = file_name[-2:]  # Extract last two digits as system name
        
        weights = pd.to_numeric(group['weight'], errors='coerce')
        
        if column == 'annotation':
            values = (group[column] == 1).astype(float)
        else:
            values = pd.to_numeric(group[column], errors='coerce')
        
        numerator = (weights * values).sum()
        denominator = weights.sum()
        
        score = numerator / denominator if denominator > 0 else 0
        if topic not in scores:
            scores[topic] = {}
        scores[topic][system] = score
    return scores

def calculate_ranking_metrics(human_scores, llm_scores):
    all_systems = list(set(human_scores.keys()) | set(llm_scores.keys()))
    
    human_score_list = [human_scores.get(system, 0) for system in all_systems]
    llm_score_list = [llm_scores.get(system, 0) for system in all_systems]
    
    ap_5 = average_precision_at_k(human_score_list, llm_score_list, 5)
    ap_10 = average_precision_at_k(human_score_list, llm_score_list, 10)
    tau, _ = stats.kendalltau(human_score_list, llm_score_list)
    pearson, _ = stats.pearsonr(human_score_list, llm_score_list)
    
    return ap_5, ap_10, tau, pearson

def average_precision_at_k(true_scores, pred_scores, k):
    paired = sorted(zip(true_scores, pred_scores), key=lambda x: x[1], reverse=True)
    true_sorted = [x[0] for x in paired][:k]
    
    precisions = []
    num_relevant = 0
    for i, score in enumerate(true_sorted, 1):
        if score > 0:
            num_relevant += 1
            precisions.append(num_relevant / i)
    
    if not precisions:
        return 0.0
    
    return sum(precisions) / min(k, len(true_scores))


def calculate_basic_metrics(df):
    # Binary metrics
    tp = ((df['annotation'] > 0) & (df['annotation'] == df['pi'])).sum()  # Matches for relevance > 0
    fp = ((df['annotation'] == 0) & (df['pi'] > 0)).sum()  # Predicted relevant but not actually relevant
    fn = ((df['annotation'] > 0) & (df['pi'] == 0)).sum()  # Actually relevant but predicted non-relevant
    tn = ((df['annotation'] == 0) & (df['pi'] == 0)).sum()  # Matches for non-relevance

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    # overlap=tp/(tp+fp+fn) if (tp+fp+fn) > 0 else 0

    # Graded overlap calculation
    # True positives for overlap: relevant levels only (> 0)
    true_positives = ((df['annotation'] > 0) & (df['annotation'] == df['pi'])).sum()
    # False predictions: Predicted relevance levels that do not match actual levels
    false_predictions = ((df['pi'] > 0) & (df['annotation'] != df['pi'])).sum()
    overlap = true_positives / (true_positives + false_predictions) if (true_positives + false_predictions) > 0 else 0

    # Graded Cohen's Kappa
    kappa = cohen_kappa_score(df['annotation'], df['pi'])

    return {
        'True Positives': tp,
        'False Positives': fp,
        'False Negatives': fn,
        'True Negatives': tn,
        'Precision': precision,
        'Recall': recall,
        "Cohen's Kappa": kappa,
        "Overlap": overlap
    }

def adjust_probabilities(df, prob_column, lower_bound, upper_bound):
    df['adjusted_prob'] = df[prob_column]
    mask = df[prob_column].between(lower_bound, upper_bound)
    df.loc[mask, 'adjusted_prob'] = df.loc[mask, 'annotation'].map({'Yes': 1.0, 'No': 0.0})
    return df