import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import load
import matplotlib.pyplot as plt
import pickle

st.set_page_config(layout="wide")

# WHat to do
# Check generic vs branded in all_features to ensure proper ordering
# In fact this hsould be done for all one_hot_encoding for proper ordering
def dd_map():
	drug_diagnosis_map = {'tanoclolol': ['G99.93',
						  'K32.86',
						  'W50.87',
						  'P29.44',
						  'V97.67',
						  'E30.01',
						  'G64.90'],
						 'oxasoted': ['U60.52',
						  'C10.29',
						  'Z98.86',
						  'I59.87',
						  'S15.62',
						  'W00.33',
						  'A13.39',
						  'P59.66',
						  'E78.98',
						  'U07.99'],
						 'cupitelol': ['Q85.91', 'W50.87', 'V97.67', 'E30.01', 'G64.90'],
						 'mamate': ['N55.01',
						  'N80.59',
						  'I38.43',
						  'C36.99',
						  'M31.63',
						  'A45.07',
						  'X00.63'],
						 'lalol': ['K32.86', 'W50.87', 'E30.01', 'G64.90', 'T44.18', 'P29.44'],
						 'foxivelule': ['U60.52',
						  'U27.71',
						  'E71.74',
						  'Q74.21',
						  'U41.19',
						  'T57.97',
						  'E11.62',
						  'C74.81',
						  'T63.88',
						  'I42.31'],
						 'tafistitrisin': ['K32.86',
						  'N27.23',
						  'N59.44',
						  'N55.01',
						  'Z95.40',
						  'B63.86',
						  'R52.40'],
						 'prazinib': ['I68.27',
						  'G51.87',
						  'U61.13',
						  'S15.62',
						  'H11.36',
						  'H26.61',
						  'A45.07'],
						 'momudobatin': ['Q72.66',
						  'C98.15',
						  'U61.13',
						  'H51.45',
						  'B03.27',
						  'X68.30',
						  'T63.88',
						  'G05.97'],
						 'cibroniudosin': ['K32.86', 'N55.01', 'Z95.40', 'N27.23', 'N59.44', 'B63.86'],
						 'rulfalol': ['K32.86',
						  'Q85.91',
						  'W50.87',
						  'V97.67',
						  'G99.93',
						  'E30.01',
						  'G64.90',
						  'T44.18',
						  'P29.44'],
						 'keglusited': ['U60.52',
						  'C10.29',
						  'S15.62',
						  'I59.87',
						  'Z98.86',
						  'W00.33',
						  'E78.98',
						  'U07.99'],
						 'pucomalol': ['Q85.91',
						  'K32.86',
						  'G99.93',
						  'W50.87',
						  'E30.01',
						  'T44.18',
						  'G64.90',
						  'P29.44',
						  'C36.69'],
						 'glycontazepelol': ['K32.86',
						  'W50.87',
						  'Q85.91',
						  'C36.69',
						  'E30.01',
						  'P29.44',
						  'G64.90'],
						 'glycogane': ['I38.43',
						  'U27.71',
						  'H60.83',
						  'Y51.55',
						  'X68.30',
						  'W45.59',
						  'J20.78',
						  'E98.09',
						  'A45.07',
						  'W74.56'],
						 'cicrochoric': ['P77.13',
						  'E49.13',
						  'G95.10',
						  'X26.32',
						  'T44.18',
						  'W40.94',
						  'G05.97',
						  'J65.04',
						  'C14.00'],
						 'satrade': ['P07.55', 'C48.92', 'V90.72', 'L07.00', 'M68.79', 'M50.24'],
						 'tovane': ['U27.71',
						  'Y11.84',
						  'Y51.55',
						  'H60.83',
						  'W45.59',
						  'H33.06',
						  'A13.39',
						  'A45.07',
						  'J20.78',
						  'E98.09',
						  'W74.56',
						  'Z04.98'],
						 'suvinicuvir': ['M42.24', 'E30.01', 'P29.44', 'Y97.47', 'C87.86'],
						 'semufolic': ['H36.57',
						  'W50.87',
						  'M42.24',
						  'P07.55',
						  'I59.87',
						  'C10.29',
						  'N48.90',
						  'W21.26',
						  'E98.09'],
						 'ratin': ['Q72.66',
						  'C98.15',
						  'U61.13',
						  'H51.45',
						  'U06.52',
						  'B03.27',
						  'X68.30',
						  'G05.97',
						  'T63.88',
						  'U97.42',
						  'C68.95'],
						 'hozirol': ['G99.93', 'O91.95', 'Q32.32', 'N48.90', 'B42.10', 'R52.40'],
						 'dienulol': ['K32.86',
						  'G99.93',
						  'V97.67',
						  'W50.87',
						  'E30.01',
						  'G64.90',
						  'T44.18',
						  'C36.69',
						  'P29.44'],
						 'hivir': ['M42.24', 'P29.44', 'E30.01', 'Y97.47', 'C87.86'],
						 'diadaric': ['Z66.42',
						  'H86.54',
						  'P77.13',
						  'E49.13',
						  'W40.94',
						  'G05.97',
						  'X26.32',
						  'T44.18',
						  'K78.87'],
						 'tocilic': ['H36.57',
						  'P07.55',
						  'C10.29',
						  'N48.90',
						  'I59.87',
						  'M42.24',
						  'E98.09'],
						 'gorol': ['G99.93', 'L85.96', 'O91.95', 'B42.10', 'N48.90', 'R52.40'],
						 'simarol': ['G99.93', 'L85.96', 'O91.95', 'N48.90', 'R52.40'],
						 'vocopirin': ['K87.68', 'Y51.55', 'A14.01', 'L07.00'],
						 'cycloxasonol': ['V80.87',
						  'W40.94',
						  'B03.27',
						  'C57.40',
						  'C74.81',
						  'Q32.32',
						  'N59.44',
						  'Q60.75'],
						 'bovirol': ['G99.93',
						  'L85.96',
						  'O91.95',
						  'B42.10',
						  'N48.90',
						  'Q32.32',
						  'R52.40'],
						 'brede': ['P07.55',
						  'M68.79',
						  'L07.00',
						  'V90.72',
						  'C48.92',
						  'C36.69',
						  'Y08.66'],
						 'nusudaric': ['Z66.42', 'E49.13', 'P77.13', 'W40.94', 'X26.32', 'G05.97'],
						 'mule': ['I68.27',
						  'U60.52',
						  'Q74.21',
						  'E71.74',
						  'T57.97',
						  'E11.62',
						  'C74.81',
						  'I42.31'],
						 'nephelilin': ['M06.85',
						  'X75.30',
						  'R33.59',
						  'Z34.94',
						  'V70.38',
						  'Y97.47',
						  'G10.16',
						  'C68.95'],
						 'antimab': ['H36.57',
						  'J12.31',
						  'T57.97',
						  'N37.61',
						  'N22.62',
						  'B84.86',
						  'L68.59',
						  'H96.75',
						  'A00.82'],
						 'plazamiglutic': ['B05.36', 'D13.25', 'Q85.91', 'G95.10', 'Z95.40', 'B03.27'],
						 'colifunene': ['I38.43',
						  'U27.71',
						  'Y51.55',
						  'H33.06',
						  'H60.83',
						  'A13.39',
						  'J20.78',
						  'W45.59',
						  'A45.07',
						  'G05.93',
						  'E98.09',
						  'W74.56',
						  'Z04.98'],
						 'vivafastat': ['Z20.23', 'L85.13', 'T84.94', 'N33.46', 'A22.87'],
						 'tugesutin': ['Q72.66',
						  'U06.52',
						  'C98.15',
						  'U61.13',
						  'X68.30',
						  'H51.45',
						  'C68.95',
						  'T63.88'],
						 'isobrovelin': ['X75.30',
						  'M06.85',
						  'H33.06',
						  'G10.16',
						  'Z34.94',
						  'Y97.47',
						  'C68.95'],
						 'hidizuzunib': ['I68.27',
						  'G51.87',
						  'U61.13',
						  'S15.62',
						  'A45.07',
						  'H11.36',
						  'H26.61'],
						 'flacelfatastat': ['Z20.23', 'L85.13', 'A22.87', 'T84.94'],
						 'choxestamenium': ['H60.83',
						  'W82.27',
						  'G51.87',
						  'Q80.97',
						  'M31.63',
						  'I42.31',
						  'M22.64'],
						 'colade': ['P07.55', 'V90.72', 'M68.79', 'L07.00', 'C48.92', 'M50.24'],
						 'sorine': ['I38.43',
						  'Y51.55',
						  'U27.71',
						  'A13.39',
						  'H33.06',
						  'W45.59',
						  'J20.78',
						  'A45.07',
						  'E98.09',
						  'G05.93',
						  'W74.56',
						  'Z04.98'],
						 'dusin': ['N27.23', 'B63.86', 'N59.44', 'N55.01', 'Z95.40', 'R52.40'],
						 'tricatripride': ['P07.55', 'V90.72', 'M68.79', 'C36.69', 'C48.92'],
						 'fazipilin': ['X75.30',
						  'R33.59',
						  'M06.85',
						  'Z34.94',
						  'H33.06',
						  'V70.38',
						  'C68.95'],
						 'tocrocin': ['Z25.06', 'L68.59', 'Z34.94', 'S78.19', 'B42.10', 'G64.90'],
						 'glulune': ['U27.71',
						  'I38.43',
						  'Y51.55',
						  'A13.39',
						  'J20.78',
						  'E98.09',
						  'W45.59',
						  'G05.93',
						  'A45.07',
						  'X68.30',
						  'A00.82',
						  'W74.56'],
						 'fumiluric': ['G95.10', 'K78.87', 'X26.32', 'W40.94', 'G05.97', 'C14.00'],
						 'kediborin': ['K87.68', 'H47.09', 'L07.00', 'A14.01', 'Y08.66'],
						 'ribosatharin': ['A14.01', 'K87.68', 'Y51.55'],
						 'todiadianic': ['B45.03', 'N55.01', 'W00.33', 'E11.62'],
						 'spifistime': ['H54.35', 'Z98.86', 'A13.39', 'E86.20', 'X68.30', 'T84.94'],
						 'nicotilin': ['M06.85', 'H33.06', 'G10.16', 'Y97.47', 'U97.42', 'C68.95'],
						 'notin': ['U06.52', 'U61.13', 'C98.15', 'B03.27', 'T63.88', 'C68.95'],
						 'gosate': ['N80.59', 'I38.43', 'M31.63', 'C36.99', 'N55.01', 'A45.07'],
						 'monemodiase': ['U75.95', 'B45.03', 'H33.06', 'L07.00'],
						 'gentipapavir': ['M42.24', 'E30.01', 'P29.44', 'Y97.47'],
						 'debome': ['H60.83', 'H54.35', 'Z98.86', 'A13.39', 'X68.30'],
						 'lixegimoric': ['X26.32',
						  'E49.13',
						  'W40.94',
						  'T44.18',
						  'K78.87',
						  'G05.97',
						  'C14.00'],
						 'sacrode': ['M68.79', 'P07.55', 'L07.00', 'V90.72', 'C48.92'],
						 'prefluflomycin': ['C14.36',
						  'I91.91',
						  'H86.54',
						  'P79.51',
						  'H89.42',
						  'Q60.75'],
						 'thiostasteglume': ['Z98.86', 'H54.35', 'A13.39', 'E86.20', 'X68.30'],
						 'cuxirin': ['H47.09', 'K87.68', 'Y51.55', 'A14.01'],
						 'flalurin': ['K87.68', 'Y51.55', 'H47.09', 'A14.01'],
						 'genetramycicin': ['Z25.06',
						  'Z34.94',
						  'L68.59',
						  'A14.01',
						  'B42.10',
						  'S78.19'],
						 'gohevitravir': ['E30.01', 'P29.44', 'Y97.47'],
						 'pheromycin': ['C14.36', 'H89.42', 'P79.51', 'W33.42', 'Q60.75'],
						 'sizubesin': ['N27.23', 'B63.86', 'N55.01', 'Z95.40', 'N59.44'],
						 'lehydrome': ['E86.20', 'H54.35', 'A13.39', 'X68.30'],
						 'masonic': ['B45.03', 'N55.01', 'E11.62'],
						 'devacin': ['Z25.06', 'Z34.94', 'A14.01', 'S78.19', 'G27.20'],
						 'pranic': ['B45.03', 'N55.01', 'E11.62'],
						 'nenizevir': ['E30.01', 'Y97.47', 'P29.44', 'C87.86']}
	diagnosis_drug_map = {'G99.93': ['tanoclolol',
						  'hozirol',
						  'gorol',
						  'simarol',
						  'bovirol',
						  'pucomalol',
						  'dienulol',
						  'rulfalol'],
						 'U60.52': ['oxasoted', 'foxivelule', 'keglusited', 'mule'],
						 'Q85.91': ['cupitelol',
						  'pucomalol',
						  'rulfalol',
						  'glycontazepelol',
						  'plazamiglutic'],
						 'N55.01': ['mamate',
						  'cibroniudosin',
						  'tafistitrisin',
						  'sizubesin',
						  'dusin',
						  'todiadianic',
						  'pranic',
						  'gosate',
						  'masonic'],
						 'K32.86': ['lalol',
						  'tafistitrisin',
						  'tanoclolol',
						  'cibroniudosin',
						  'rulfalol',
						  'pucomalol',
						  'glycontazepelol',
						  'dienulol'],
						 'U27.71': ['foxivelule',
						  'tovane',
						  'glycogane',
						  'colifunene',
						  'glulune',
						  'sorine'],
						 'I68.27': ['prazinib', 'mule', 'hidizuzunib'],
						 'Q72.66': ['momudobatin', 'ratin', 'tugesutin'],
						 'G51.87': ['prazinib', 'hidizuzunib', 'choxestamenium'],
						 'I38.43': ['glycogane',
						  'colifunene',
						  'sorine',
						  'mamate',
						  'glulune',
						  'gosate'],
						 'P77.13': ['cicrochoric', 'nusudaric', 'diadaric'],
						 'P07.55': ['satrade',
						  'brede',
						  'tocilic',
						  'colade',
						  'tricatripride',
						  'semufolic',
						  'sacrode'],
						 'C98.15': ['momudobatin', 'ratin', 'tugesutin', 'notin'],
						 'E71.74': ['foxivelule', 'mule'],
						 'M42.24': ['suvinicuvir', 'hivir', 'semufolic', 'gentipapavir', 'tocilic'],
						 'H36.57': ['semufolic', 'tocilic', 'antimab'],
						 'W50.87': ['cupitelol',
						  'glycontazepelol',
						  'lalol',
						  'rulfalol',
						  'tanoclolol',
						  'semufolic',
						  'pucomalol',
						  'dienulol'],
						 'Z66.42': ['diadaric', 'nusudaric'],
						 'Y11.84': ['tovane'],
						 'C10.29': ['oxasoted', 'tocilic', 'keglusited', 'semufolic'],
						 'N80.59': ['mamate', 'gosate'],
						 'K87.68': ['vocopirin', 'kediborin', 'ribosatharin', 'flalurin', 'cuxirin'],
						 'Q74.21': ['foxivelule', 'mule'],
						 'V80.87': ['cycloxasonol'],
						 'V97.67': ['rulfalol', 'cupitelol', 'dienulol', 'tanoclolol'],
						 'M06.85': ['nephelilin', 'fazipilin', 'nicotilin', 'isobrovelin'],
						 'B05.36': ['plazamiglutic'],
						 'D13.25': ['plazamiglutic'],
						 'Z20.23': ['vivafastat', 'flacelfatastat'],
						 'U61.13': ['momudobatin',
						  'ratin',
						  'prazinib',
						  'notin',
						  'hidizuzunib',
						  'tugesutin'],
						 'Z98.86': ['oxasoted',
						  'spifistime',
						  'thiostasteglume',
						  'keglusited',
						  'debome'],
						 'X75.30': ['isobrovelin', 'fazipilin', 'nephelilin'],
						 'H51.45': ['ratin', 'momudobatin', 'tugesutin'],
						 'Y51.55': ['tovane',
						  'colifunene',
						  'sorine',
						  'ribosatharin',
						  'glulune',
						  'glycogane',
						  'cuxirin',
						  'vocopirin',
						  'flalurin'],
						 'W40.94': ['cycloxasonol',
						  'lixegimoric',
						  'fumiluric',
						  'diadaric',
						  'nusudaric',
						  'cicrochoric'],
						 'E30.01': ['rulfalol',
						  'gohevitravir',
						  'suvinicuvir',
						  'gentipapavir',
						  'pucomalol',
						  'cupitelol',
						  'lalol',
						  'glycontazepelol',
						  'tanoclolol',
						  'hivir',
						  'dienulol',
						  'nenizevir'],
						 'H60.83': ['choxestamenium', 'tovane', 'colifunene', 'debome', 'glycogane'],
						 'N27.23': ['dusin', 'cibroniudosin', 'tafistitrisin', 'sizubesin'],
						 'E49.13': ['nusudaric', 'cicrochoric', 'lixegimoric', 'diadaric'],
						 'Z25.06': ['tocrocin', 'genetramycicin', 'devacin'],
						 'W45.59': ['tovane', 'colifunene', 'sorine', 'glulune', 'glycogane'],
						 'G95.10': ['fumiluric', 'plazamiglutic', 'cicrochoric'],
						 'U06.52': ['tugesutin', 'notin', 'ratin'],
						 'A14.01': ['ribosatharin',
						  'genetramycicin',
						  'devacin',
						  'vocopirin',
						  'cuxirin',
						  'flalurin',
						  'kediborin'],
						 'R33.59': ['fazipilin', 'nephelilin'],
						 'H47.09': ['kediborin', 'cuxirin', 'flalurin'],
						 'L85.96': ['gorol', 'simarol', 'bovirol'],
						 'U41.19': ['foxivelule'],
						 'B45.03': ['todiadianic', 'masonic', 'monemodiase', 'pranic'],
						 'H54.35': ['spifistime', 'thiostasteglume', 'debome', 'lehydrome'],
						 'P29.44': ['tanoclolol',
						  'glycontazepelol',
						  'hivir',
						  'gohevitravir',
						  'suvinicuvir',
						  'lalol',
						  'dienulol',
						  'rulfalol',
						  'gentipapavir',
						  'pucomalol',
						  'nenizevir'],
						 'I59.87': ['oxasoted', 'tocilic', 'keglusited', 'semufolic'],
						 'B03.27': ['cycloxasonol', 'momudobatin', 'notin', 'ratin', 'plazamiglutic'],
						 'H33.06': ['colifunene',
						  'tovane',
						  'sorine',
						  'isobrovelin',
						  'nicotilin',
						  'monemodiase',
						  'fazipilin'],
						 'C48.92': ['satrade', 'brede', 'colade', 'sacrode', 'tricatripride'],
						 'J12.31': ['antimab'],
						 'U75.95': ['monemodiase'],
						 'A13.39': ['tovane',
						  'colifunene',
						  'glulune',
						  'sorine',
						  'oxasoted',
						  'thiostasteglume',
						  'spifistime',
						  'lehydrome',
						  'debome'],
						 'N48.90': ['tocilic', 'bovirol', 'simarol', 'gorol', 'hozirol', 'semufolic'],
						 'C57.40': ['cycloxasonol'],
						 'T57.97': ['foxivelule', 'antimab', 'mule'],
						 'O91.95': ['gorol', 'hozirol', 'simarol', 'bovirol'],
						 'X26.32': ['lixegimoric',
						  'cicrochoric',
						  'fumiluric',
						  'nusudaric',
						  'diadaric'],
						 'Z95.40': ['cibroniudosin',
						  'tafistitrisin',
						  'plazamiglutic',
						  'sizubesin',
						  'dusin'],
						 'M68.79': ['sacrode', 'brede', 'colade', 'satrade', 'tricatripride'],
						 'S15.62': ['prazinib', 'oxasoted', 'keglusited', 'hidizuzunib'],
						 'C14.36': ['prefluflomycin', 'pheromycin'],
						 'B42.10': ['gorol', 'bovirol', 'hozirol', 'tocrocin', 'genetramycicin'],
						 'L68.59': ['tocrocin', 'antimab', 'genetramycicin'],
						 'J20.78': ['colifunene', 'glulune', 'tovane', 'sorine', 'glycogane'],
						 'C74.81': ['cycloxasonol', 'foxivelule', 'mule'],
						 'E11.62': ['foxivelule', 'pranic', 'mule', 'todiadianic', 'masonic'],
						 'L85.13': ['flacelfatastat', 'vivafastat'],
						 'V90.72': ['colade', 'satrade', 'brede', 'sacrode', 'tricatripride'],
						 'H86.54': ['diadaric', 'prefluflomycin'],
						 'N37.61': ['antimab'],
						 'Q32.32': ['hozirol', 'bovirol', 'cycloxasonol'],
						 'W82.27': ['choxestamenium'],
						 'N22.62': ['antimab'],
						 'I91.91': ['prefluflomycin'],
						 'W00.33': ['oxasoted', 'keglusited', 'todiadianic'],
						 'B84.86': ['antimab'],
						 'E86.20': ['lehydrome', 'thiostasteglume', 'spifistime'],
						 'G10.16': ['isobrovelin', 'nicotilin', 'nephelilin'],
						 'K78.87': ['fumiluric', 'lixegimoric', 'diadaric'],
						 'N59.44': ['tafistitrisin',
						  'cibroniudosin',
						  'dusin',
						  'cycloxasonol',
						  'sizubesin'],
						 'L07.00': ['brede',
						  'colade',
						  'satrade',
						  'sacrode',
						  'vocopirin',
						  'kediborin',
						  'monemodiase'],
						 'M31.63': ['gosate', 'mamate', 'choxestamenium'],
						 'Z34.94': ['devacin',
						  'genetramycicin',
						  'tocrocin',
						  'fazipilin',
						  'isobrovelin',
						  'nephelilin'],
						 'P59.66': ['oxasoted'],
						 'S78.19': ['tocrocin', 'devacin', 'genetramycicin'],
						 'Q80.97': ['choxestamenium'],
						 'C36.69': ['glycontazepelol',
						  'dienulol',
						  'pucomalol',
						  'brede',
						  'tricatripride'],
						 'P79.51': ['prefluflomycin', 'pheromycin'],
						 'C36.99': ['mamate', 'gosate'],
						 'T44.18': ['pucomalol',
						  'cicrochoric',
						  'lalol',
						  'dienulol',
						  'lixegimoric',
						  'rulfalol',
						  'diadaric'],
						 'B63.86': ['dusin', 'cibroniudosin', 'sizubesin', 'tafistitrisin'],
						 'G64.90': ['rulfalol',
						  'dienulol',
						  'lalol',
						  'cupitelol',
						  'pucomalol',
						  'glycontazepelol',
						  'tanoclolol',
						  'tocrocin'],
						 'E98.09': ['glulune',
						  'colifunene',
						  'tovane',
						  'sorine',
						  'tocilic',
						  'semufolic',
						  'glycogane'],
						 'H96.75': ['antimab'],
						 'H89.42': ['pheromycin', 'prefluflomycin'],
						 'X68.30': ['tugesutin',
						  'glycogane',
						  'glulune',
						  'thiostasteglume',
						  'ratin',
						  'momudobatin',
						  'spifistime',
						  'lehydrome',
						  'debome'],
						 'Y97.47': ['hivir',
						  'nenizevir',
						  'gohevitravir',
						  'isobrovelin',
						  'suvinicuvir',
						  'nicotilin',
						  'nephelilin',
						  'gentipapavir'],
						 'R52.40': ['bovirol',
						  'gorol',
						  'tafistitrisin',
						  'hozirol',
						  'simarol',
						  'dusin'],
						 'H11.36': ['prazinib', 'hidizuzunib'],
						 'H26.61': ['prazinib', 'hidizuzunib'],
						 'G05.97': ['diadaric',
						  'nusudaric',
						  'cicrochoric',
						  'ratin',
						  'momudobatin',
						  'lixegimoric',
						  'fumiluric'],
						 'A45.07': ['prazinib',
						  'colifunene',
						  'tovane',
						  'glulune',
						  'sorine',
						  'hidizuzunib',
						  'mamate',
						  'glycogane',
						  'gosate'],
						 'W33.42': ['pheromycin'],
						 'G05.93': ['glulune', 'colifunene', 'sorine'],
						 'A00.82': ['glulune', 'antimab'],
						 'V70.38': ['nephelilin', 'fazipilin'],
						 'W21.26': ['semufolic'],
						 'T63.88': ['foxivelule', 'ratin', 'momudobatin', 'notin', 'tugesutin'],
						 'I42.31': ['choxestamenium', 'foxivelule', 'mule'],
						 'A22.87': ['flacelfatastat', 'vivafastat'],
						 'M22.64': ['choxestamenium'],
						 'W74.56': ['glulune', 'colifunene', 'tovane', 'sorine', 'glycogane'],
						 'M50.24': ['colade', 'satrade'],
						 'Y08.66': ['brede', 'kediborin'],
						 'E78.98': ['keglusited', 'oxasoted'],
						 'U07.99': ['keglusited', 'oxasoted'],
						 'U97.42': ['ratin', 'nicotilin'],
						 'Q60.75': ['pheromycin', 'prefluflomycin', 'cycloxasonol'],
						 'T84.94': ['flacelfatastat', 'spifistime', 'vivafastat'],
						 'C68.95': ['nephelilin',
						  'nicotilin',
						  'tugesutin',
						  'isobrovelin',
						  'ratin',
						  'notin',
						  'fazipilin'],
						 'G27.20': ['devacin'],
						 'J65.04': ['cicrochoric'],
						 'C87.86': ['hivir', 'suvinicuvir', 'nenizevir'],
						 'Z04.98': ['tovane', 'sorine', 'colifunene'],
						 'C14.00': ['lixegimoric', 'fumiluric', 'cicrochoric'],
						 'X00.63': ['mamate'],
						 'N33.46': ['vivafastat']}
	return drug_diagnosis_map, diagnosis_drug_map

drug_diagnosis_map, diagnosis_drug_map = dd_map()
#@st.cache
def sns_plot(drug_type, df):

	# Looks like the dataframe are stored as strings
	df['branded'] = [float(val) for val in df.branded.values]
	df['generic'] = [float(val) for val in df.generic.values]
	df.sort_values(by=['branded'], ascending=True, inplace=True)
	#df.sort_values(by=['generic'], ascending=True, inplace=True)
	x = np.arange(len(df))
	y = np.linspace(df.branded.values[0], df.branded.values[-1], len(df)).tolist()
	#y = np.linspace(df.generic.values[0], df.generic.values[-1], len(df)).tolist()


	fig = plt.figure(figsize=(10,5))
	plt.scatter(x, df.generic.values, color='b', s=50)
	plt.scatter(x, df.branded.values, color='r', s=50)
	plt.title('Drugs similar to ' + drug_type +  ' based on diagnosis', fontsize=15)
	plt.xlabel('Drugs',fontsize=12)
	plt.ylabel('Predicted copay',fontsize=12)
	plt.xticks(x, df.drugs.values, rotation = 45,  color='k', fontsize=8)
	plt.yticks(y, color='k', fontsize=8)
	plt.legend(labels=['generic', 'branded'], fontsize=15)
	return fig

@st.cache
def predict_copay(model, features):

	# All one_hot_encoded values in training code
	all_features = ["drug_group_branded", "drug_group_generic",  "drug_type_antimab", \
					"drug_type_bovirol", "drug_type_brede", "drug_type_choxestamenium",\
					"drug_type_cibroniudosin", "drug_type_cicrochoric", "drug_type_colade",\
					"drug_type_colifunene", "drug_type_cupitelol", "drug_type_cuxirin", \
					"drug_type_cycloxasonol", "drug_type_debome", "drug_type_devacin",\
					"drug_type_diadaric", "drug_type_dienulol", "drug_type_dusin",\
					"drug_type_fazipilin", "drug_type_flacelfatastat", \
					"drug_type_flalurin", "drug_type_foxivelule", \
					"drug_type_fumiluric", "drug_type_genetramycicin", \
					"drug_type_gentipapavir", "drug_type_glulune", \
					"drug_type_glycogane", "drug_type_glycontazepelol", \
					"drug_type_gohevitravir", "drug_type_gorol", "drug_type_gosate",\
					"drug_type_hidizuzunib", "drug_type_hivir", "drug_type_hozirol", \
					"drug_type_isobrovelin", "drug_type_kediborin", "drug_type_keglusited",\
					"drug_type_lalol", "drug_type_lehydrome", "drug_type_lixegimoric", \
					"drug_type_mamate", "drug_type_masonic", "drug_type_momudobatin",\
					"drug_type_monemodiase", "drug_type_mule", "drug_type_nenizevir", \
					"drug_type_nephelilin", "drug_type_nicotilin", "drug_type_notin", \
					"drug_type_nusudaric", "drug_type_oxasoted", "drug_type_pheromycin",\
					"drug_type_plazamiglutic", "drug_type_pranic", "drug_type_prazinib",\
					"drug_type_prefluflomycin", "drug_type_pucomalol", "drug_type_ratin",\
					"drug_type_ribosatharin", "drug_type_rulfalol", "drug_type_sacrode",\
					"drug_type_satrade", "drug_type_semufolic", "drug_type_simarol", \
					"drug_type_sizubesin", "drug_type_sorine", "drug_type_spifistime", \
					"drug_type_suvinicuvir", "drug_type_tafistitrisin", "drug_type_tanoclolol",\
					"drug_type_thiostasteglume", "drug_type_tocilic", "drug_type_tocrocin", \
					"drug_type_todiadianic", "drug_type_tovane", "drug_type_tricatripride", \
					"drug_type_tugesutin", "drug_type_vivafastat", "drug_type_vocopirin", "pcn_1UQC", \
					"pcn_2TIC", "pcn_327CKV", "pcn_393U", "pcn_3O71UTS", "pcn_3Y5ZW0", "pcn_6ZGS97C", \
					"pcn_7THOQ5", "pcn_9C5MOR3", "pcn_9D24", "pcn_9FU70", "pcn_AZUO5U", "pcn_BIZF", "pcn_BZ22Z2",\
					"pcn_CG3ZWQ", "pcn_CS8580", "pcn_DY4B", "pcn_FX2Z", "pcn_IF448", "pcn_J5DT8", "pcn_K5KDJ7G",\
					"pcn_KB38N", "pcn_KBOSN", "pcn_MQWH09H", "pcn_MSCXSG", "pcn_N098KI", "pcn_NC7EN", "pcn_NG4CS",\
					"pcn_OO0E", "pcn_P4LC", "pcn_RAM3J", "pcn_RB7UU", "pcn_REGLCC", "pcn_RM0HB", "pcn_S76J7V6",\
					"pcn_T17LNK", "pcn_T52GV", "pcn_TAZ5W", "pcn_TPJD", "pcn_ULM7G", "pcn_W1LW9Y", "pcn_W7L3", "pcn_WM6A",\
					"pcn_XH4T3", "pcn_YFVIA", "pcn_YICC41", "pcn_YL5CMT", "pcn_ZQPX", "pcn_pcn_0", "pcn_pcn_1", "pcn_pcn_10",\
					"pcn_pcn_11", "pcn_pcn_12", "pcn_pcn_13", "pcn_pcn_14", "pcn_pcn_2", "pcn_pcn_3", "pcn_pcn_4", "pcn_pcn_5", \
					"pcn_pcn_6", "pcn_pcn_7", "pcn_pcn_8", "pcn_pcn_9"]


	# # all_unknown pcn_values groupings
	pcn_val = {'group_0': '1UQC',
			 '52H8KH0F83K': 'pcn_2',
			 '6BYJBW': 'KB38N',
			 'ZX2QUWR': 'pcn_13',
			 'group_10': 'S76J7V6',
			 'IOEAN1DWVV3Y': '327CKV',
			 '1CAHL': 'pcn_1',
			 'HO8HUGL': 'pcn_7',
			 'group_6': 'CS8580',
			 'STGRDKR1J5RD': 'pcn_10',
			 'DGLGRYP': 'MSCXSG',
			 'group_9': 'NC7EN',
			 'I4UYEP84W3': 'pcn_8',
			 'group_8': 'KBOSN',
			 'group_1': '3O71UTS',
			 'KZWQDIHCLLHD1': 'ZQPX',
			 'EVD4X5': 'T52GV',
			 '6SP1DG': 'N098KI',
			 'group_7': 'DY4B',
			 'IGN6JL34H37D': 'AZUO5U',
			 'group_2': '3Y5ZW0',
			 'RS5RB3YA': 'RB7UU',
			 'SJVO3GXUURRGO': 'RM0HB',
			 'QK6BI1N61': 'BIZF',
			 '7DUPMODV0': 'RAM3J',
			 'T51T6V2E8L': 'TAZ5W',
			 'L9QZA': 'WM6A',
			 'IX6P0': 'J5DT8',
			 'AJK5MZ25T9IA': 'YFVIA',
			 '0OGKQ': 'W1LW9Y',
			 'HPVSQW7M8': 'OO0E',
			 'ZOYKF0N5NEO': 'BZ22Z2',
			 'O19XSLDEFB': 'P4LC',
			 'group_3': '6ZGS97C',
			 'VC81HUO7ZD': '9D24',
			 'YY6B1J4E8KJ3': 'pcn_12',
			 'DYGBI610ZY': 'pcn_6',
			 'Z01MLD4I': 'T17LNK',
			 'group_11': 'TPJD',
			 '7LL04USF': 'pcn_4',
			 'BH2Q8B3GY2GAV': 'REGLCC',
			 'MP3IQ': 'YICC41',
			 '9R3Z3QKDF3': 'pcn_5',
			 'GOM8K0': 'XH4T3',
			 'group_4': '7THOQ5',
			 'group_5': 'CG3ZWQ',
			 'FZPLF4O6FD': 'IF448',
			 'GQIGYFQQ2WGH': 'NG4CS',
			 'XK8RM5E75ZW': '2TIC',
			 '0TZ9XYJZJH': 'pcn_0',
			 '7Q756WMLLV25X': 'MQWH09H',
			 'XY5GQQ9': '393U',
			 'S2QKZ0OFNWS6X': '9C5MOR3',
			 'group_13': 'YL5CMT',
			 'U19J4RVCA': '9FU70',
			 'group_12': 'ULM7G',
			 'RGVK1': 'pcn_9',
			 'V96T9QL5': 'W7L3',
			 '77MAJF66DGD': 'pcn_3',
			 'Unknown': 'pcn_14',
			 'TFZOR5R49': 'pcn_11',
			 '1N5IRQ': 'K5KDJ7G',
			 'OD99VAJGWV': 'FX2Z'}

	# if pcn_values is unknown, check the group class to obtain pcn_value
	if features[2] == 'Unknown':
		features[2] = pcn_val[features[3]]

	all_drugs = []
	all_diagnosis = []
	for diagnosis in drug_diagnosis_map[features[1]]:
		all_drugs.extend(diagnosis_drug_map[diagnosis])
		all_diagnosis.append(diagnosis)

	all_drugs = np.unique(all_drugs)
	branded_features = []
	for drug in all_drugs:

		# Now get one_hot_encoded correspondence of inputed values
		feature = ['drug_group_branded', 'drug_type_' + drug, 'pcn_' + features[2]]

		# All values in one_hot_encoded are zeroes except of inputed values
		in_features = np.zeros(len(all_features))

		# Get the indices corresponding to inputed values and change to 1
		indices = [i for i, x in enumerate(all_features) for val in feature if x == val ]
		in_features[indices] = 1
		branded_features.append(in_features)
	df_branded = pd.DataFrame(branded_features, columns = all_features)
	branded_drugs_pay = model.predict(df_branded)

	generic_features = []
	for drug in all_drugs:
		# Now get one_hot_encoded correspondence of inputed values
		feature = ['drug_group_generic', 'drug_type_' + drug, 'pcn_' + features[2]]

		# All values in one_hot_encoded are zeroes except of inputed values
		in_features = np.zeros(len(all_features))

		# Get the indices corresponding to inputed values and change to 1
		indices = [i for i, x in enumerate(all_features) for val in feature if x == val ]
		in_features[indices] = 1
		generic_features.append(in_features)
	df_generic = pd.DataFrame(generic_features, columns = all_features)
	generic_drugs_pay = model.predict(df_generic)

	df = pd.DataFrame(np.array([all_drugs, branded_drugs_pay, generic_drugs_pay]).T, columns=['drugs', 'branded', 'generic'])
	if features[0] == 'branded':
		estimate = df.loc[df.drugs == features[1]].branded.values[0]
	else:
		estimate = df.loc[df.drugs == features[1]].generic.values[0]

	return estimate, all_diagnosis, df

# @st.cache
# def predict_copay(model, features):
#
# 	# All one_hot_encoded values in training code
# 	all_features = [ "branded", "generic", "327CKV", "pcn_12", "pcn_1", "CS8580", "RM0HB", "S76J7V6", "KB38N", "pcn_8", "pcn_2", "pcn_7", "pcn_10", "7THOQ5", "3O71UTS", "MSCXSG", "N098KI", "pcn_6", "pcn_9", "NC7EN", "BZ22Z2", "WM6A", "pcn_4", "pcn_13", "T52GV", "1UQC", "BIZF", "3Y5ZW0", "REGLCC", "TPJD", "RB7UU", "J5DT8", "pcn_5", "IF448", "9FU70", "DY4B", "NG4CS", "ZQPX", "9D24", "AZUO5U", "TAZ5W", "W1LW9Y", "YFVIA", "6ZGS97C", "T17LNK", "9C5MOR3", "2TIC", "XH4T3", "OO0E", "KBOSN", "YICC41", "pcn_14", "pcn_11", "YL5CMT", "W7L3", "pcn_0", "393U", "MQWH09H", "K5KDJ7G", "CG3ZWQ", "ULM7G", "RAM3J", "P4LC", "FX2Z", "pcn_3", "dienulol", "nicotilin", "cibroniudosin", "hidizuzunib", "plazamiglutic", "foxivelule", "mule", "prazinib", "pucomalol", "gorol", "fazipilin", "rulfalol", "bovirol", "antimab", "oxasoted", "vocopirin", "fumiluric", "cupitelol", "diadaric", "lalol", "tocilic", "isobrovelin", "keglusited", "vivafastat", "hivir", "sorine", "cuxirin", "glulune", "momudobatin", "ribosatharin", "ratin", "tovane", "glycontazepelol", "tanoclolol", "colade", "suvinicuvir", "semufolic", "nephelilin", "choxestamenium", "brede", "pranic", "nusudaric", "cicrochoric", "hozirol", "notin", "tocrocin", "tugesutin", "todiadianic", "tricatripride", "glycogane", "monemodiase", "gosate", "simarol", "lixegimoric", "flalurin", "colifunene", "sacrode", "satrade", "tafistitrisin", "cycloxasonol", "gentipapavir", "sizubesin", "mamate", "devacin", "thiostasteglume", "prefluflomycin", "flacelfatastat", "genetramycicin", "lehydrome", "spifistime", "kediborin", "pheromycin", "debome", "dusin", "gohevitravir", "masonic", "nenizevir"]
#
#
# 	# # all_unknown pcn_values groupings
# 	pcn_val = {'group_0': '1UQC',
# 			 '52H8KH0F83K': 'pcn_2',
# 			 '6BYJBW': 'KB38N',
# 			 'ZX2QUWR': 'pcn_13',
# 			 'group_10': 'S76J7V6',
# 			 'IOEAN1DWVV3Y': '327CKV',
# 			 '1CAHL': 'pcn_1',
# 			 'HO8HUGL': 'pcn_7',
# 			 'group_6': 'CS8580',
# 			 'STGRDKR1J5RD': 'pcn_10',
# 			 'DGLGRYP': 'MSCXSG',
# 			 'group_9': 'NC7EN',
# 			 'I4UYEP84W3': 'pcn_8',
# 			 'group_8': 'KBOSN',
# 			 'group_1': '3O71UTS',
# 			 'KZWQDIHCLLHD1': 'ZQPX',
# 			 'EVD4X5': 'T52GV',
# 			 '6SP1DG': 'N098KI',
# 			 'group_7': 'DY4B',
# 			 'IGN6JL34H37D': 'AZUO5U',
# 			 'group_2': '3Y5ZW0',
# 			 'RS5RB3YA': 'RB7UU',
# 			 'SJVO3GXUURRGO': 'RM0HB',
# 			 'QK6BI1N61': 'BIZF',
# 			 '7DUPMODV0': 'RAM3J',
# 			 'T51T6V2E8L': 'TAZ5W',
# 			 'L9QZA': 'WM6A',
# 			 'IX6P0': 'J5DT8',
# 			 'AJK5MZ25T9IA': 'YFVIA',
# 			 '0OGKQ': 'W1LW9Y',
# 			 'HPVSQW7M8': 'OO0E',
# 			 'ZOYKF0N5NEO': 'BZ22Z2',
# 			 'O19XSLDEFB': 'P4LC',
# 			 'group_3': '6ZGS97C',
# 			 'VC81HUO7ZD': '9D24',
# 			 'YY6B1J4E8KJ3': 'pcn_12',
# 			 'DYGBI610ZY': 'pcn_6',
# 			 'Z01MLD4I': 'T17LNK',
# 			 'group_11': 'TPJD',
# 			 '7LL04USF': 'pcn_4',
# 			 'BH2Q8B3GY2GAV': 'REGLCC',
# 			 'MP3IQ': 'YICC41',
# 			 '9R3Z3QKDF3': 'pcn_5',
# 			 'GOM8K0': 'XH4T3',
# 			 'group_4': '7THOQ5',
# 			 'group_5': 'CG3ZWQ',
# 			 'FZPLF4O6FD': 'IF448',
# 			 'GQIGYFQQ2WGH': 'NG4CS',
# 			 'XK8RM5E75ZW': '2TIC',
# 			 '0TZ9XYJZJH': 'pcn_0',
# 			 '7Q756WMLLV25X': 'MQWH09H',
# 			 'XY5GQQ9': '393U',
# 			 'S2QKZ0OFNWS6X': '9C5MOR3',
# 			 'group_13': 'YL5CMT',
# 			 'U19J4RVCA': '9FU70',
# 			 'group_12': 'ULM7G',
# 			 'RGVK1': 'pcn_9',
# 			 'V96T9QL5': 'W7L3',
# 			 '77MAJF66DGD': 'pcn_3',
# 			 'Unknown': 'pcn_14',
# 			 'TFZOR5R49': 'pcn_11',
# 			 '1N5IRQ': 'K5KDJ7G',
# 			 'OD99VAJGWV': 'FX2Z'}
# 	#st.write(features)
# 	# if pcn_values is unknown, check the group class to obtain pcn_value
# 	if features[2] == 'Unknown':
# 		features[2] = pcn_val[features[3]]
#
# 	all_drugs = []
# 	all_diagnosis = []
# 	for diagnosis in drug_diagnosis_map[features[1]]:
# 		all_drugs.extend(diagnosis_drug_map[diagnosis])
# 		all_diagnosis.append(diagnosis)
#
# 	all_drugs = np.unique(all_drugs)
# 	branded_features = []
# 	for drug in all_drugs:
#
# 		# Now get one_hot_encoded correspondence of inputed values
# 		feature = ['branded', features[2], drug]
#
# 		# All values in one_hot_encoded are zeroes except of inputed values
# 		in_features = np.zeros(len(all_features))
#
# 		# Get the indices corresponding to inputed values and change to 1
# 		indices = [i for i, x in enumerate(all_features) for val in feature if x == val ]
# 		in_features[indices] = 1
# 		branded_features.append(in_features)
#
# 	branded_drugs_pay = model.predict(branded_features)
#
# 	generic_features = []
# 	for drug in all_drugs:
# 		# Now get one_hot_encoded correspondence of inputed values
# 		feature = ['generic', features[2], drug]
#
# 		# All values in one_hot_encoded are zeroes except of inputed values
# 		in_features = np.zeros(len(all_features))
#
# 		# Get the indices corresponding to inputed values and change to 1
# 		indices = [i for i, x in enumerate(all_features) for val in feature if x == val ]
# 		in_features[indices] = 1
# 		generic_features.append(in_features)
# 	generic_drugs_pay = model.predict(generic_features)
#
# 	df = pd.DataFrame(np.array([all_drugs, branded_drugs_pay, generic_drugs_pay]).T, columns=['drugs', 'branded', 'generic'])
# 	#st.write(df)
# 	if features[0] == 'branded':
# 		estimate = df.loc[df.drugs == features[1]].branded.values[0]
# 	else:
# 		estimate = df.loc[df.drugs == features[1]].generic.values[0]
#
# 	return estimate, all_diagnosis, df


def convert_prob(probs, threshold):
    vals = []

    thresh_2 = (threshold + 1)/2
    for prob in probs:
        if prob < threshold:
            vals.append('non-covered')
        elif threshold <= prob and prob < thresh_2:
            vals.append('non-preferred')
        else:
            vals.append('preferred')
    return vals

@st.cache
def predict_formulary(model, features, threshold):

	# All one_hot_encoded values in training code
	all_features = ["drug_group_branded", "drug_group_generic",  "drug_type_antimab", \
					"drug_type_bovirol", "drug_type_brede", "drug_type_choxestamenium",\
					"drug_type_cibroniudosin", "drug_type_cicrochoric", "drug_type_colade",\
					"drug_type_colifunene", "drug_type_cupitelol", "drug_type_cuxirin", \
					"drug_type_cycloxasonol", "drug_type_debome", "drug_type_devacin",\
					"drug_type_diadaric", "drug_type_dienulol", "drug_type_dusin",\
					"drug_type_fazipilin", "drug_type_flacelfatastat", \
					"drug_type_flalurin", "drug_type_foxivelule", \
					"drug_type_fumiluric", "drug_type_genetramycicin", \
					"drug_type_gentipapavir", "drug_type_glulune", \
					"drug_type_glycogane", "drug_type_glycontazepelol", \
					"drug_type_gohevitravir", "drug_type_gorol", "drug_type_gosate",\
					"drug_type_hidizuzunib", "drug_type_hivir", "drug_type_hozirol", \
					"drug_type_isobrovelin", "drug_type_kediborin", "drug_type_keglusited",\
					"drug_type_lalol", "drug_type_lehydrome", "drug_type_lixegimoric", \
					"drug_type_mamate", "drug_type_masonic", "drug_type_momudobatin",\
					"drug_type_monemodiase", "drug_type_mule", "drug_type_nenizevir", \
					"drug_type_nephelilin", "drug_type_nicotilin", "drug_type_notin", \
					"drug_type_nusudaric", "drug_type_oxasoted", "drug_type_pheromycin",\
					"drug_type_plazamiglutic", "drug_type_pranic", "drug_type_prazinib",\
					"drug_type_prefluflomycin", "drug_type_pucomalol", "drug_type_ratin",\
					"drug_type_ribosatharin", "drug_type_rulfalol", "drug_type_sacrode",\
					"drug_type_satrade", "drug_type_semufolic", "drug_type_simarol", \
					"drug_type_sizubesin", "drug_type_sorine", "drug_type_spifistime", \
					"drug_type_suvinicuvir", "drug_type_tafistitrisin", "drug_type_tanoclolol",\
					"drug_type_thiostasteglume", "drug_type_tocilic", "drug_type_tocrocin", \
					"drug_type_todiadianic", "drug_type_tovane", "drug_type_tricatripride", \
					"drug_type_tugesutin", "drug_type_vivafastat", "drug_type_vocopirin", "pcn_1UQC", \
					"pcn_2TIC", "pcn_327CKV", "pcn_393U", "pcn_3O71UTS", "pcn_3Y5ZW0", "pcn_6ZGS97C", \
					"pcn_7THOQ5", "pcn_9C5MOR3", "pcn_9D24", "pcn_9FU70", "pcn_AZUO5U", "pcn_BIZF", "pcn_BZ22Z2",\
					"pcn_CG3ZWQ", "pcn_CS8580", "pcn_DY4B", "pcn_FX2Z", "pcn_IF448", "pcn_J5DT8", "pcn_K5KDJ7G",\
					"pcn_KB38N", "pcn_KBOSN", "pcn_MQWH09H", "pcn_MSCXSG", "pcn_N098KI", "pcn_NC7EN", "pcn_NG4CS",\
					"pcn_OO0E", "pcn_P4LC", "pcn_RAM3J", "pcn_RB7UU", "pcn_REGLCC", "pcn_RM0HB", "pcn_S76J7V6",\
					"pcn_T17LNK", "pcn_T52GV", "pcn_TAZ5W", "pcn_TPJD", "pcn_ULM7G", "pcn_W1LW9Y", "pcn_W7L3", "pcn_WM6A",\
					"pcn_XH4T3", "pcn_YFVIA", "pcn_YICC41", "pcn_YL5CMT", "pcn_ZQPX", "pcn_pcn_0", "pcn_pcn_1", "pcn_pcn_10",\
					"pcn_pcn_11", "pcn_pcn_12", "pcn_pcn_13", "pcn_pcn_14", "pcn_pcn_2", "pcn_pcn_3", "pcn_pcn_4", "pcn_pcn_5", \
					"pcn_pcn_6", "pcn_pcn_7", "pcn_pcn_8", "pcn_pcn_9"]


	# # all_unknown pcn_values groupings
	pcn_val = {'group_0': '1UQC',
			 '52H8KH0F83K': 'pcn_2',
			 '6BYJBW': 'KB38N',
			 'ZX2QUWR': 'pcn_13',
			 'group_10': 'S76J7V6',
			 'IOEAN1DWVV3Y': '327CKV',
			 '1CAHL': 'pcn_1',
			 'HO8HUGL': 'pcn_7',
			 'group_6': 'CS8580',
			 'STGRDKR1J5RD': 'pcn_10',
			 'DGLGRYP': 'MSCXSG',
			 'group_9': 'NC7EN',
			 'I4UYEP84W3': 'pcn_8',
			 'group_8': 'KBOSN',
			 'group_1': '3O71UTS',
			 'KZWQDIHCLLHD1': 'ZQPX',
			 'EVD4X5': 'T52GV',
			 '6SP1DG': 'N098KI',
			 'group_7': 'DY4B',
			 'IGN6JL34H37D': 'AZUO5U',
			 'group_2': '3Y5ZW0',
			 'RS5RB3YA': 'RB7UU',
			 'SJVO3GXUURRGO': 'RM0HB',
			 'QK6BI1N61': 'BIZF',
			 '7DUPMODV0': 'RAM3J',
			 'T51T6V2E8L': 'TAZ5W',
			 'L9QZA': 'WM6A',
			 'IX6P0': 'J5DT8',
			 'AJK5MZ25T9IA': 'YFVIA',
			 '0OGKQ': 'W1LW9Y',
			 'HPVSQW7M8': 'OO0E',
			 'ZOYKF0N5NEO': 'BZ22Z2',
			 'O19XSLDEFB': 'P4LC',
			 'group_3': '6ZGS97C',
			 'VC81HUO7ZD': '9D24',
			 'YY6B1J4E8KJ3': 'pcn_12',
			 'DYGBI610ZY': 'pcn_6',
			 'Z01MLD4I': 'T17LNK',
			 'group_11': 'TPJD',
			 '7LL04USF': 'pcn_4',
			 'BH2Q8B3GY2GAV': 'REGLCC',
			 'MP3IQ': 'YICC41',
			 '9R3Z3QKDF3': 'pcn_5',
			 'GOM8K0': 'XH4T3',
			 'group_4': '7THOQ5',
			 'group_5': 'CG3ZWQ',
			 'FZPLF4O6FD': 'IF448',
			 'GQIGYFQQ2WGH': 'NG4CS',
			 'XK8RM5E75ZW': '2TIC',
			 '0TZ9XYJZJH': 'pcn_0',
			 '7Q756WMLLV25X': 'MQWH09H',
			 'XY5GQQ9': '393U',
			 'S2QKZ0OFNWS6X': '9C5MOR3',
			 'group_13': 'YL5CMT',
			 'U19J4RVCA': '9FU70',
			 'group_12': 'ULM7G',
			 'RGVK1': 'pcn_9',
			 'V96T9QL5': 'W7L3',
			 '77MAJF66DGD': 'pcn_3',
			 'Unknown': 'pcn_14',
			 'TFZOR5R49': 'pcn_11',
			 '1N5IRQ': 'K5KDJ7G',
			 'OD99VAJGWV': 'FX2Z'}

	# if pcn_values is unknown, check the group class to obtain pcn_value
	if features[2] == 'Unknown':
		features[2] = pcn_val[features[3]]

	all_drugs = []
	all_diagnosis = []
	for diagnosis in drug_diagnosis_map[features[1]]:
		all_drugs.extend(diagnosis_drug_map[diagnosis])
		all_diagnosis.append(diagnosis)

	all_drugs = np.unique(all_drugs)
	#branded_drugs_pay = {}
	branded_features = []
	for drug in all_drugs:
		# Now get one_hot_encoded correspondence of inputed values
		feature = ['drug_group_branded', 'drug_type_' + drug, 'pcn_' + features[2]]

		# All values in one_hot_encoded are zeroes except of inputed values
		in_features = np.zeros(len(all_features))

		# Get the indices corresponding to inputed values and change to 1
		indices = [i for i, x in enumerate(all_features) for val in feature if x == val ]
		in_features[indices] = 1
		branded_features.append(in_features)
	df_branded = pd.DataFrame(branded_features, columns = all_features)
	branded_drugs_pay = convert_prob(model.predict_proba(df_branded)[:,0], threshold)

	#generic_drugs_pay = {}
	generic_features = []
	for drug in all_drugs:
		# Now get one_hot_encoded correspondence of inputed values
		feature = ['drug_group_generic', 'drug_type_' + drug, 'pcn_' + features[2]]

		# All values in one_hot_encoded are zeroes except of inputed values
		in_features = np.zeros(len(all_features))

		# Get the indices corresponding to inputed values and change to 1
		indices = [i for i, x in enumerate(all_features) for val in feature if x == val ]
		in_features[indices] = 1
		generic_features.append(in_features)
	df_generic = pd.DataFrame(generic_features, columns = all_features)
	generic_drugs_pay = convert_prob(model.predict_proba(df_generic)[:,0], threshold)

	df = pd.DataFrame(np.array([all_drugs, branded_drugs_pay, generic_drugs_pay]).T, columns=['drugs', 'branded', 'generic'])

	if features[0] == 'branded':
		estimate = df.loc[df.drugs == features[1]].branded.values[0]
	else:
		estimate = df.loc[df.drugs == features[1]].generic.values[0]

	return estimate,  df

def main():

	activities = [ "ABOUT", "COPAYMENT PREDICTION", "FORMULARY STATUS"]
	choice = st.sidebar.selectbox("SELECT ACTIVITY",activities)
	model_pay = load('dtr_model.joblib')
	#model_pay = pickle.load(open('forest_model.sav', 'rb'))
	model_formulary, best_threshold  = load('best_model_formulary.joblib')
	drug_group = ['branded', 'generic']
	drug_type = ['tanoclolol', 'oxasoted', 'cupitelol', 'mamate', 'lalol',
				   'foxivelule', 'tafistitrisin', 'prazinib', 'momudobatin',
				   'cibroniudosin', 'rulfalol', 'keglusited', 'pucomalol',
				   'glycontazepelol', 'glycogane', 'cicrochoric', 'satrade', 'tovane',
				   'suvinicuvir', 'semufolic', 'ratin', 'hozirol', 'dienulol',
				   'hivir', 'diadaric', 'tocilic', 'gorol', 'simarol', 'vocopirin',
				   'cycloxasonol', 'bovirol', 'brede', 'nusudaric', 'mule',
				   'nephelilin', 'antimab', 'plazamiglutic', 'colifunene',
				   'vivafastat', 'tugesutin', 'isobrovelin', 'hidizuzunib',
				   'flacelfatastat', 'choxestamenium', 'colade', 'sorine', 'dusin',
				   'tricatripride', 'fazipilin', 'tocrocin', 'glulune', 'fumiluric',
				   'kediborin', 'ribosatharin', 'todiadianic', 'spifistime',
				   'nicotilin', 'notin', 'gosate', 'monemodiase', 'gentipapavir',
				   'debome', 'lixegimoric', 'sacrode', 'prefluflomycin',
				   'thiostasteglume', 'cuxirin', 'flalurin', 'genetramycicin',
				   'gohevitravir', 'pheromycin', 'sizubesin', 'lehydrome', 'masonic',
				   'devacin', 'pranic', 'nenizevir']
	pcn = ['1UQC', 'Unknown', 'KB38N', 'S76J7V6', '327CKV', 'CS8580',
	        'MSCXSG', 'NC7EN', 'KBOSN', '3O71UTS', 'ZQPX', 'T52GV', 'N098KI',
	        'DY4B', 'AZUO5U', 'RM0HB', 'BIZF', 'RAM3J', 'TAZ5W', 'WM6A',
	        'J5DT8', 'W1LW9Y', 'OO0E', 'BZ22Z2', 'RB7UU', 'P4LC', '6ZGS97C',
	        'T17LNK', 'TPJD', 'REGLCC', 'YICC41', 'XH4T3', '7THOQ5', 'YFVIA',
	        'CG3ZWQ', 'IF448', 'NG4CS', '2TIC', '9D24', 'MQWH09H', '393U',
	        'YL5CMT', '9FU70', '3Y5ZW0', 'ULM7G', 'W7L3', '9C5MOR3', 'K5KDJ7G',
	        'FX2Z']
	group = ['Unknown', '52H8KH0F83K', '6BYJBW', 'ZX2QUWR', 'IOEAN1DWVV3Y', '1CAHL',
		   'HO8HUGL', 'STGRDKR1J5RD', 'DGLGRYP', 'I4UYEP84W3',
		   'KZWQDIHCLLHD1', 'EVD4X5', '6SP1DG', 'IGN6JL34H37D', 'RS5RB3YA',
		   'SJVO3GXUURRGO', 'QK6BI1N61', '7DUPMODV0', 'T51T6V2E8L', 'L9QZA',
		   'IX6P0', 'AJK5MZ25T9IA', '0OGKQ', 'HPVSQW7M8', 'ZOYKF0N5NEO',
		   'O19XSLDEFB', 'VC81HUO7ZD', 'YY6B1J4E8KJ3', 'DYGBI610ZY',
		   'Z01MLD4I', '7LL04USF', 'BH2Q8B3GY2GAV', 'MP3IQ', '9R3Z3QKDF3',
		   'GOM8K0', 'FZPLF4O6FD', 'GQIGYFQQ2WGH', 'XK8RM5E75ZW',
		   '0TZ9XYJZJH', '7Q756WMLLV25X', 'XY5GQQ9', 'S2QKZ0OFNWS6X',
		   'U19J4RVCA', 'RGVK1', 'V96T9QL5', '77MAJF66DGD', 'TFZOR5R49',
		   '1N5IRQ', 'OD99VAJGWV']

	drug_group.sort(reverse=False)
	pcn.sort(reverse=False)
	drug_type.sort(reverse=False)
	group.sort(reverse=False)

	if choice == "COPAYMENT PREDICTION":
		st.title("CoverMyMeds App")
		st.text("Built with Streamlit and Scikit-learn")

		st.subheader("Select features from left menu")

		drug_group_feature = st.sidebar.selectbox("drug_brand", drug_group)
		drug_type_feature = st.sidebar.selectbox("drug_type", drug_type)
		pcn_feature = st.sidebar.selectbox("pcn", pcn)

		group_feature = ''
		if pcn_feature == 'Unknown':
			group_feature = st.sidebar.selectbox("group", group)

		if st.button("Predict"):
			features = [drug_group_feature, drug_type_feature, pcn_feature, group_feature]
			estimate, diagnosis, df = predict_copay(model_pay, features)
			strs = 'Estimated patient pay for '  + drug_group_feature + '_' + drug_type_feature  + ' with pcn(' +  pcn_feature + '): $' +  str(round(float(estimate), 2))
			st.text(strs)
			st.text('Drug (' + drug_type_feature + ') has been administered for the following diagnosis:  ' + ', '.join(diagnosis))
			st.pyplot(sns_plot(drug_type_feature, df))


	if choice == "FORMULARY STATUS":
		st.title("CoverMyMeds App")
		st.text("Built with Streamlit and Scikit-learn")

		st.subheader("Select features from left menu")

		drug_group_feature = st.sidebar.selectbox("drug_brand", drug_group)
		drug_type_feature = st.sidebar.selectbox("drug_type", drug_type)
		pcn_feature = st.sidebar.selectbox("pcn", pcn)

		group_feature = ''
		if pcn_feature == 'Unknown':
			group_feature = st.sidebar.selectbox("group", group)

		if st.button("Predict"):
			features = [drug_group_feature, drug_type_feature, pcn_feature, group_feature]
			estimate,  df = predict_formulary(model_formulary, features, best_threshold)
			strs = 'Formulary status for '  + drug_group_feature + '_' + drug_type_feature  + ' with pcn (' +  pcn_feature + '): ' +  estimate
			st.text(strs)
			if estimate != 'preferred':
				drugs_g = df[df.generic == 'preferred'].drugs.values.tolist()
				drugs_b = df[df.branded == 'preferred'].drugs.values.tolist()
				len_b = len(drugs_b)
				len_g = len(drugs_g)
				if len_b < len_g:
					len1 = len_g - len_b
					drugs_b.extend([' ']* len1)
				else:
					len1 = len_b - len_g
					drugs_g.extend([' ']* len1)

				st.text('Alternative drugs (with preferred formulary status) to ' + drug_group_feature + '_' + drug_type_feature + ' with similar diagnosis: ' )
				drugs_df = pd.DataFrame(np.array([drugs_b, drugs_g]).T, columns = ['Branded', 'Generic'])
				st.write(drugs_df)

	if choice == "ABOUT":
			st.subheader("About CoverMyMeds App")
			st.markdown("Built with Streamlit and Scikit-learn by May22-Galileo (The Erdos Institute BootCamp)")
	#Add a feedback section in the sidebar
	st.sidebar.title(' ') #Used to create some space between the filter widget and the comments section
	st.sidebar.markdown(' ') #Used to create some space between the filter widget and the comments section
	st.sidebar.subheader('Please help us improve the app!')
	with st.sidebar.form(key='columns_in_form',clear_on_submit=True): #set clear_on_submit=True so that the form will be reset/cleared once it's submitted
	    rating=st.slider("Please rate the app", min_value=0, max_value=5, value=3,help='Drag the slider to rate the app. This is a 0-5 rating scale where 5 is the highest rating')
	    text=st.text_input(label='Please leave your feedback here:')
	    submitted = st.form_submit_button('Submit')
	    if submitted:
	      st.write('Thanks for your feedback!')
	      st.markdown('Your Rating:')
	      st.markdown(rating)
	      st.markdown('Your Feedback:')
	      st.markdown(text)


if __name__ == '__main__':
		main()
