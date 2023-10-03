
if 'Collins_Wilkie_Armadale_1864' in self.doc_path:
    text = text.replace('scoundrel?', 'scoundrel ')

if 'Dickens_Charles_The-Pickwick-Papers_1836' in self.doc_path:
    text = text.replace('I was;3', 'I was; 3')  
    text = text.replace('P.V.P.M.P.C.,1', 'P.V.P.M.P.C.') 
    text = text.replace('G.C.M.P.C.,2', 'G.C.M.P.C.')

if 'Haggard_H-Rider_King-Solomons-Mines_1885' in self.doc_path:            
    text = text.replace(' – Editor.', '')
    text = text.replace('Kukuanaland.1', 'Kukuanaland.')


if 'Morrison_Arthur_A-Child-of-the-Jago_1896' in self.doc_path:
    text = text.replace('(which = watch stealer)', '')

if 'La-Roche_Sophie_Fraeulein-von-Sternheim_1771' in self.doc_path:
    text = text.replace('[', '')


if 'Fielding_Henry_Joseph-Andrews_1742' in self.doc_path:
    text = text.replace(' Footnote 5: ', '')

if 'Scott_Walter_The-Heart-of-Midlothian_1818' in self.doc_path:
    text = text.replace('â€˜the', 'the')
    text = text.replace('[', '')
    text = text.replace(']', '')

if 'Alexis_Willibald_Schloss-Avalon_1826' in self.doc_path:
    text = text.replace(':|:', '')



if 'Unger_Friederike-Helene_Julchen-Gruenthal_1784' in self.doc_path:
    text = text.replace('@', 'li')



if 'La-Roche_Sophie_Rosaliens-Briefe_1780' in self.doc_path:
    text = text.replace('[«] [Anschlußfehler in der Vorlage]' ,'')
    text = text.replace('[Anschlußfehler in der Vorlage, M.L.]' ,'')
    text = text.replace('@', '')
    text = text.replace('[»]', '')

if 'Hoffmann_ETA_Berganza_1814' in self.doc_path:
    text = text.replace('. Anmerk. des Verlegers [C. F. Kunz].' ,'')

if 'Eliot_George_Romola_1862' in self.doc_path:
    text = text.replace('(See note at the end.)','')

if 'Mundt_Theodor_Madonna_1835' in self.doc_path:
    long_string = '(1801-1882), "Wlasta" (1829), großes böhmischnationales Heldengedicht. – Anm.d.Hrsg.'
    text = text.replace(long_string, '')

if 'Stifter_Adalbert_Die-Narrenburg_1843' in self.doc_path:
    text = text.replace('*) [*)Alpenstock]' ,'')

if 'Oliphant_Margaret_The-Perpetual-Curate_1864' in self.doc_path:
    text = text.replace('a-visiting[' ,'a-visiting')

if 'Burney_Frances_The-Wanderer_1814' in self.doc_path:
    text = text.replace('averseness]' ,'averseness')

if 'Scott_Walter_The-Abbot_1820' in self.doc_path:
    text = text.replace('Footnote:', '')
    text = text.replace('[footnote: Pancakes]', '')


if 'May_Karl_Das-Waldroeschen_1883' in self.doc_path:
    text = text.replace('– Anmerkung des Verfassers.]', '')
    text = text.replace('[', '')
    
if 'Stevenson-Grift_Robert-Louis-Fanny-van-de_The-Dynamiter_1885' in self.doc_path:
    text = text.replace('\with','with')

if 'Brooke_Frances_Lady-Julia-Mandeville_1763' in self.doc_path:
    text = text.replace('=Your', '')

if 'Kirchbach_Wolfgang_Das-Leben-auf-der-Walze_1892' in self.doc_path:
    text = text.replace('^', '')
    text = text.replace('Katzenkopp = Schlosser', '')

if 'Schloegl_Friedrich_Wiener-Blut_1873' in self.doc_path:
    text = text.replace('^', '')
    text = text.replace('=', '-')

if 'Baldwin_Louisa_Sir-Nigel-Otterburnes-Case_1895' in self.doc_path:
    text = text.replace('~', 'v')

if 'Baldwin_Louisa_Many-Waters-Cannot-Quench-Love_1895' in self.doc_path:
    text = text.replace('th~', 'the')

if 'Baldwin_Louisa_The-Empty-Picture-Frame_1895' in self.doc_path:
    text = text.replace('~', '')
    text = text.replace('original]', 'original')

if 'Forrester_Andrew_The-Female-Detective_1864' in self.doc_path:
    text = text.replace('~', '')
    text = text.replace('{', '')

if 'Gaskell_Elizabeth_Mary-Barton_1848' in self.doc_path:
    text = text.replace('Footnote 40:', '')

if 'Lever_Charles_Confessions-of-Harry-Lorrequer_1837' in self.doc_path:
    text = text.replace('FOOTNOTE:', '')

if 'Holcroft_Thomas_Anna-St-Ives_1792' in self.doc_path:
    text = text.replace('[Footnote 1: Omitted.]', '')
    text = text.replace('Footnote 1: ', '')


if 'Marryat_Frederick_The-Kings-Own_1830' in self.doc_path:
    text = text.replace('(see note 1)', '')

if 'Dickens_Charles_Oliver-Twist_1837' in self.doc_path:
    text = text.replace('Footnote:', '')

if 'Moerike_Eduard_Das-Stuttgarter-Hutzelmaennchen_1853' in self.doc_path:
    text = text.replace('Koloczaer Kodex altdeutscher Ged., hrsg. von Mailath usw., S. 232.', '')

if 'Defoe_Daniel_Journal-of-the-Plague-Year_1722' in self.doc_path:
    text = text.replace('[Footnote in the original.]', '')
    text = text.replace('[Footnotes in the original.]', '')
    text = text.replace('{*}', '')
    text = text.replace('*', '')

if 'Baldwin_Louisa_The-Ticking-of-the-Clock_1895' in self.doc_path:
    text = text.replace('B]ewitt', 'Blewitt')

if 'Swift_Jonathan_Gullivers-Travels_1726' in self.doc_path:
    text = text.replace('[As given in the original edition.]', '')

if 'Tressell_Robert_The-Ragged-Trousered-Philanthropists_1914' in self.doc_path:
    text = text.replace('}', '')

if 'Auerbach_Berthold_Die-Frau-Professorin_1846' in self.doc_path:
    text = text.replace('Fußnoten1 ', '')

if 'Owenson_Sydney_The-Wild-Irish-Girl_1806' in self.doc_path:
    text = text.replace('Ã¦', 'ae')
    text = text.replace('=', '')

if 'Baldwin_Louisa_The-Shadow-on-the-Blind_1895' in self.doc_path:
    text = text.replace('o~', 'or')

if 'Collins_Wilkie_The-Woman-in-White_1859' in self.doc_path:
    text = text.replace('(see Sermon XXIX. in the Collection by the late Rev. Samuel Michelson, M.A.)', '')
    text = text.replace('[', '')
    text = text.replace(']', '')

if 'Gutzkow_Karl_Briefe-einers-Narren-und-einer-Naerrin_1832' in self.doc_path:
    long_string = r'[Lat.; zusammengezogen aus (ita) me Dius Fidius (iuvet) = So wahr mir (der treue) Gott helfe! Bei Gott! Wahrhaftig!]  (Ein römischer Schwur, der wohl schwer zu übersetzen, aber nicht unerklärlich ist.)'
    text = text.replace(long_string, '')

if 'Paul_Jean_Flegeljahre_1804' in self.doc_path:
    text = text.replace('(^-^-)', '')
    text = text.replace('(^^-^)', '')
    text = text.replace('(--^^)', '')

if 'Freytag_Gustav_Die-Ahnen_1872' in self.doc_path:
    text = text.replace('~', 'v')

if 'Bronte_Charlotte_Shirley_1849' in self.doc_path:

if 'Hogg_James_The-Sheperds-Calender_1829' in self.doc_path:
    text = text.replace('Pharmacop[oe]ia', 'Pharmacopoeia')

if 'Butler_Samuel_The-Way-of-all-Flesh_1903' in self.doc_path:
    text = text.replace('[wick-ed', 'wicked')
    text = text.replace('[Music score]', '')
    text = text.replace(']', '')







