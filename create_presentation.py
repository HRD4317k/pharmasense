"""Generates the 14-slide PharmaSense Checkpoint 2 evaluation presentation.
Adheres strictly to Checkpoint 1 branding, colors, 16:9 layout, text-heavy content,
plain language (no formulas), and includes speaker notes for every slide.
Refined to eliminate dead space, ensure perfect contrast, and maximize readability.
"""
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

OUTPUT_FILE = Path(__file__).resolve().parent / "PharmaSense_Checkpoint_2.pptx"

# ----------------- Color Palette (from Checkpoint 1 & app.py) -----------------
DARK_BG    = RGBColor(0x0B, 0x1D, 0x24)  # #0B1D24 deep slate navy
LIGHT_BG   = RGBColor(0xF6, 0xF9, 0xFA)  # #F6F9FA clean crisp light background
WHITE      = RGBColor(0xFF, 0xFF, 0xFF)
CARD_BG    = RGBColor(0xFF, 0xFF, 0xFF)  # Card container
CARD_ALT   = RGBColor(0xEE, 0xF6, 0xF8)  # Light mint-tint card
BORDER_COL = RGBColor(0xCF, 0xE2, 0xE6)  # Subtle card border

TEAL       = RGBColor(0x0E, 0x7C, 0x86)  # #0E7C86 primary brand teal
MINT       = RGBColor(0x2D, 0xD4, 0xBF)  # #2DD4BF vibrant mint accent
DARK_INK   = RGBColor(0x12, 0x30, 0x3A)  # #12303A deep title navy
TEXT_MUTED = RGBColor(0x56, 0x70, 0x7A)  # #56707A subtitle/caption grey
TEXT_BODY  = RGBColor(0x24, 0x38, 0x42)  # #243842 clear dark body text
TEXT_LIGHT = RGBColor(0xDC, 0xEB, 0xEE)  # Light text on dark background
DARK_CARD_BG = RGBColor(0x12, 0x29, 0x32) # Dark card container
DARK_CARD_BORDER = RGBColor(0x1E, 0x48, 0x54)

FONT_HEADING = "Segoe UI"
FONT_BODY    = "Segoe UI"

def set_slide_background(slide, color):
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    return bg

def add_header(slide, category, title, subtitle=None):
    # Left accent strip
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.24), Inches(7.5))
    accent.fill.solid()
    accent.fill.fore_color.rgb = TEAL
    accent.line.fill.background()

    # Top category tracker
    tb_cat = slide.shapes.add_textbox(Inches(0.75), Inches(0.32), Inches(11.8), Inches(0.32))
    tf_cat = tb_cat.text_frame
    tf_cat.word_wrap = True
    tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
    p_cat = tf_cat.paragraphs[0]
    p_cat.text = category.upper()
    p_cat.font.name = FONT_HEADING
    p_cat.font.size = Pt(10.5)
    p_cat.font.bold = True
    p_cat.font.color.rgb = TEAL

    # Main slide title
    tb_title = slide.shapes.add_textbox(Inches(0.75), Inches(0.62), Inches(11.8), Inches(0.52))
    tf_title = tb_title.text_frame
    tf_title.word_wrap = True
    tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
    p_title = tf_title.paragraphs[0]
    p_title.text = title
    p_title.font.name = FONT_HEADING
    p_title.font.size = Pt(21)
    p_title.font.bold = True
    p_title.font.color.rgb = DARK_INK

    # Optional subtitle
    top_offset = Inches(1.18)
    if subtitle:
        tb_sub = slide.shapes.add_textbox(Inches(0.75), Inches(1.14), Inches(11.8), Inches(0.32))
        tf_sub = tb_sub.text_frame
        tf_sub.word_wrap = True
        tf_sub.margin_left = tf_sub.margin_right = tf_sub.margin_top = tf_sub.margin_bottom = 0
        p_sub = tf_sub.paragraphs[0]
        p_sub.text = subtitle
        p_sub.font.name = FONT_BODY
        p_sub.font.size = Pt(11.5)
        p_sub.font.color.rgb = TEXT_MUTED
        top_offset = Inches(1.52)

    # Thin separator line
    line = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.75), top_offset, Inches(11.8), Inches(0.015))
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER_COL
    line.line.fill.background()

    return top_offset + Inches(0.16)

def add_card(slide, left, top, width, height, title=None, bg_color=CARD_BG, border_color=BORDER_COL, title_color=DARK_INK):
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = bg_color
    box.line.color.rgb = border_color
    box.line.width = Pt(1.2)
    
    tb = slide.shapes.add_textbox(left + Inches(0.24), top + Inches(0.20), width - Inches(0.48), height - Inches(0.40))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    
    if title:
        p0 = tf.paragraphs[0]
        p0.text = title
        p0.font.name = FONT_HEADING
        p0.font.size = Pt(13)
        p0.font.bold = True
        p0.font.color.rgb = title_color
        p0.space_after = Pt(8)
        
    return tf

def add_bullet_item(tf, heading, body, font_size=10.2, space_after=8, head_color=DARK_INK, body_color=TEXT_BODY):
    p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
    p.space_after = Pt(space_after)
    p.line_spacing = 1.15
    
    if heading:
        r_head = p.add_run()
        r_head.text = heading + " — "
        r_head.font.name = FONT_BODY
        r_head.font.size = Pt(font_size)
        r_head.font.bold = True
        r_head.font.color.rgb = head_color
    
    r_body = p.add_run()
    r_body.text = body
    r_body.font.name = FONT_BODY
    r_body.font.size = Pt(font_size)
    r_body.font.bold = False
    r_body.font.color.rgb = body_color

def set_speaker_notes(slide, notes_text):
    notes = slide.notes_slide
    tf = notes.notes_text_frame
    tf.text = notes_text

# ==============================================================================
# SLIDE BUILDERS
# ==============================================================================

def build_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    # --------------------------------------------------------------------------
    # SLIDE 1: Title Slide
    # --------------------------------------------------------------------------
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1, DARK_BG)
    
    acc1 = s1.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), Inches(7.5))
    acc1.fill.solid()
    acc1.fill.fore_color.rgb = MINT
    acc1.line.fill.background()

    tb1_tag = s1.shapes.add_textbox(Inches(1.0), Inches(0.9), Inches(11.0), Inches(0.4))
    tf1_tag = tb1_tag.text_frame
    p1_tag = tf1_tag.paragraphs[0]
    p1_tag.text = "CS61061 · DATA ANALYTICS · TRACK 2 · CHECKPOINT 2 EVALUATION"
    p1_tag.font.name = FONT_HEADING
    p1_tag.font.size = Pt(11.5)
    p1_tag.font.bold = True
    p1_tag.font.color.rgb = MINT

    tb1_t = s1.shapes.add_textbox(Inches(1.0), Inches(1.35), Inches(11.0), Inches(1.6))
    tf1_t = tb1_t.text_frame
    tf1_t.word_wrap = True
    p1_t = tf1_t.paragraphs[0]
    p1_t.text = "PharmaSense"
    p1_t.font.name = FONT_HEADING
    p1_t.font.size = Pt(46)
    p1_t.font.bold = True
    p1_t.font.color.rgb = WHITE

    p1_sub = tf1_t.add_paragraph()
    p1_sub.text = "A Multi-Gate Polypharmacy Pharmacovigilance Platform"
    p1_sub.font.name = FONT_HEADING
    p1_sub.font.size = Pt(22)
    p1_sub.font.bold = True
    p1_sub.font.color.rgb = MINT
    p1_sub.space_before = Pt(4)

    tb1_d = s1.shapes.add_textbox(Inches(1.0), Inches(3.2), Inches(11.0), Inches(1.4))
    tf1_d = tb1_d.text_frame
    tf1_d.word_wrap = True
    p1_d = tf1_d.paragraphs[0]
    p1_d.text = (
        "Mining spontaneous FDA adverse-event reports to uncover genuine drug combination safety signals, "
        "filtering reporting noise and indications, and resolving clinical blame across single drugs, pairs, "
        "and 3-drug combinations with exact statistical attribution."
    )
    p1_d.font.name = FONT_BODY
    p1_d.font.size = Pt(13)
    p1_d.font.color.rgb = TEXT_LIGHT

    b1_card = s1.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(4.85), Inches(11.0), Inches(1.9))
    b1_card.fill.solid()
    b1_card.fill.fore_color.rgb = RGBColor(0x13, 0x2A, 0x33)
    b1_card.line.color.rgb = TEAL
    b1_card.line.width = Pt(1.5)
    
    tb1_meta = s1.shapes.add_textbox(Inches(1.3), Inches(5.02), Inches(10.4), Inches(1.6))
    tf1_meta = tb1_meta.text_frame
    
    p_pres = tf1_meta.paragraphs[0]
    p_pres.text = "Presenter: Rishabh Dehariya  ·  Roll No: 23IM10028"
    p_pres.font.name = FONT_HEADING
    p_pres.font.size = Pt(14)
    p_pres.font.bold = True
    p_pres.font.color.rgb = WHITE
    
    p_dept = tf1_meta.add_paragraph()
    p_dept.text = "Department of Industrial and Systems Engineering  ·  Indian Institute of Technology Kharagpur"
    p_dept.font.name = FONT_BODY
    p_dept.font.size = Pt(12)
    p_dept.font.color.rgb = TEXT_LIGHT
    p_dept.space_before = Pt(4)

    p_sup = tf1_meta.add_paragraph()
    p_sup.text = "Project Supervisor: Prof. Sarada Prasad Sarmah"
    p_sup.font.name = FONT_BODY
    p_sup.font.size = Pt(12.5)
    p_sup.font.bold = True
    p_sup.font.color.rgb = MINT
    p_sup.space_before = Pt(4)

    set_speaker_notes(s1,
        "Good morning members of the evaluation committee. Today I present Checkpoint 2 of PharmaSense, "
        "a multi-gate pharmacovigilance platform built under the supervision of Prof. Sarada Prasad Sarmah in the "
        "Department of Industrial and Systems Engineering. PharmaSense addresses polypharmacy safety by mining "
        "millions of FDA reports to systematically differentiate true multi-drug interactions from single-drug side effects."
    )

    # --------------------------------------------------------------------------
    # SLIDE 2: Recap of Checkpoint 1
    # --------------------------------------------------------------------------
    s2 = prs.slides.add_slide(blank_layout)
    set_slide_background(s2, LIGHT_BG)
    top2 = add_header(s2, "01 / Recap of Checkpoint 1",
                      "Foundational Commitment & Roadmap Alignment",
                      "How Checkpoint 1 commitments evolved into a working multi-gate analytical pipeline")

    tf2_l = add_card(s2, Inches(0.75), top2, Inches(5.75), Inches(5.35), "Checkpoint 1 Commitments & Scope")
    add_bullet_item(tf2_l, "Identified the Pharmacovigilance Gap",
                    "Public dashboards for the 20M+ FAERS reports only allow lookup; they do not mine multi-drug combinations or surface hidden interactions.", space_after=10)
    add_bullet_item(tf2_l, "Targeted the Underserved Middle",
                    "Independent researchers, pharmacy colleges, and hospital safety boards lack affordable tools, as enterprise safety software costs $50k–$200k/year.", space_after=10)
    add_bullet_item(tf2_l, "Committed to 3-Drug Polypharmacy",
                    "Pledged to extend surveillance beyond single drugs and pairs to 3-drug combinations, where elderly and polymedicated patients face highest risk.", space_after=10)
    add_bullet_item(tf2_l, "Outlined a 4-Phase Delivery Roadmap",
                    "Scheduled Phase 1 foundation and EDA, Phase 2 association mining and disproportionality, Phase 3 temporal signal growth, and Phase 4 deployment.", space_after=10)
    add_bullet_item(tf2_l, "Architectural Foundation Defined",
                    "Committed to columnar storage using DuckDB, Python analytical algorithms, and a searchable web dashboard to enable interactive exploration.", space_after=10)

    tf2_r = add_card(s2, Inches(6.8), top2, Inches(5.75), Inches(5.35), "Where We Stand Today (Checkpoint 2 Reality)")
    add_bullet_item(tf2_r, "Execution of Planned Phase 2",
                    "Successfully delivered the Association Rule Mining (ARM) engine and disproportionality scoring framework committed in Checkpoint 1.", space_after=10)
    add_bullet_item(tf2_r, "Diagnosed Critical Statistical Inflation",
                    "Initial naive disproportionality tests produced wildly inflated risk values in the hundreds, forcing an immediate overhaul of our cleaning pipeline.", space_after=10)
    add_bullet_item(tf2_r, "Formulated Multi-Gate Safeguards",
                    "Replaced single-threshold screening with a rigorous multi-gate verification pipeline that screens out false alarms at each statistical stage.", space_after=10)
    add_bullet_item(tf2_r, "Resolved Attribution Ambiguity",
                    "Engineered an automated clinical attribution tree that separates true 3-drug synergies from bystander drugs and dual-drug interactions.", space_after=10)
    add_bullet_item(tf2_r, "Runnable Local Product Delivered",
                    "All backend mining, DuckDB database loading, predictive ML models, and the Streamlit dashboard are fully integrated and running on localhost.", space_after=10)

    set_speaker_notes(s2,
        "In Checkpoint 1, we set out to build an accessible pharmacovigilance platform focused on 3-drug combinations, "
        "serving hospital pharmacists and researchers priced out of enterprise tools. In this evaluation, we demonstrate "
        "that our association mining engine is fully built and operational. Crucially, we solved major real-world statistical "
        "challenges around data inflation by establishing a rigorous multi-gate verification architecture."
    )

    # --------------------------------------------------------------------------
    # SLIDE 3: Problem and Opportunity
    # --------------------------------------------------------------------------
    s3 = prs.slides.add_slide(blank_layout)
    set_slide_background(s3, LIGHT_BG)
    top3 = add_header(s3, "02 / Problem & Clinical Opportunity",
                      "The Severe Blindspot in Multi-Drug Surveillance",
                      "Why modern healthcare needs combination-specific adverse event pattern mining")

    tf3_l = add_card(s3, Inches(0.75), top3, Inches(5.75), Inches(5.35), "The Growing Polypharmacy Crisis")
    add_bullet_item(tf3_l, "Ubiquity in Aging Populations",
                    "Over 40% of adults aged 65 and older take 5 or more prescription drugs simultaneously, creating unpredictable pharmacodynamic interactions.", space_after=10)
    add_bullet_item(tf3_l, "Pre-Approval Clinical Trial Blindspot",
                    "Clinical trials evaluate medications in isolation or tightly controlled pairs; polymedicated patients are routinely excluded from pre-market cohorts.", space_after=10)
    add_bullet_item(tf3_l, "Siloed Package Labeling",
                    "FDA package inserts list adverse reactions drug by drug, providing zero guidance on multi-drug cocktails taken in daily practice.", space_after=10)
    add_bullet_item(tf3_l, "Escalating Hospitalization Burden",
                    "Adverse drug events cause up to 7% of emergency hospital admissions, driving billions in preventable healthcare and readmission expenses.", space_after=10)
    add_bullet_item(tf3_l, "Unreported Real-World Signals",
                    "Complex 3-drug adverse effects remain buried in spontaneous reporting databases because no automated system isolates them at scale.", space_after=10)

    tf3_r = add_card(s3, Inches(6.8), top3, Inches(5.75), Inches(5.35), "Stakeholders Left Underserved")
    add_bullet_item(tf3_r, "Clinical & Hospital Pharmacists",
                    "Face complex patient medication regimens daily without digital tools to verify whether an acute reaction is an emerging combination hazard.", space_after=10)
    add_bullet_item(tf3_r, "Academic & Clinical Researchers",
                    "Forced to perform slow, manual SQL queries on raw FAERS dumps, lacking tools for systematic multi-drug hypothesis generation.", space_after=10)
    add_bullet_item(tf3_r, "Regulatory Safety Surveillance Teams",
                    "Overwhelmed by millions of raw reports where single-drug blame is confounded by companion medications and administrative reporting noise.", space_after=10)
    add_bullet_item(tf3_r, "Small & Mid-Sized Generic Pharma",
                    "Priced out of enterprise pharmacovigilance platforms ($50k-$200k/year), leaving them without accessible post-market safety intelligence.", space_after=10)
    add_bullet_item(tf3_r, "The Market Opportunity",
                    "PharmaSense provides a lightweight, highly accurate web platform that turns raw spontaneous reports into actionable, plain-language signals.", space_after=10)

    set_speaker_notes(s3,
        "Polypharmacy represents one of the largest safety gaps in modern medicine. When a patient takes multiple drugs, "
        "pre-approval trials offer zero guidance because polymedicated patients were systematically excluded from trials. "
        "Hospital pharmacists and independent researchers currently have no affordable tools to systematically detect combination "
        "risks, leading to preventable hospitalizations and unmanaged clinical interactions."
    )

    # --------------------------------------------------------------------------
    # SLIDE 4: Product Vision (Startup Angle)
    # --------------------------------------------------------------------------
    s4 = prs.slides.add_slide(blank_layout)
    set_slide_background(s4, LIGHT_BG)
    top4 = add_header(s4, "03 / Product Vision & Commercialization",
                      "Developing a Scalable Pharmacovigilance Product",
                      "Translating complex safety intelligence into a sustainable, accessible SaaS platform")

    tf4_l = add_card(s4, Inches(0.75), top4, Inches(5.75), Inches(5.35), "Target Users & Core Value Proposition")
    add_bullet_item(tf4_l, "Distinct Customer Personas",
                    "Designed for hospital pharmacy boards, clinical toxicologists, academic pharmacy departments, and pharmacovigilance officers in mid-sized pharma.", space_after=10)
    add_bullet_item(tf4_l, "Plain-Language Clinical Intelligence",
                    "Translates intimidating statistical odds into direct clinical verdicts: stating immediately whether a reaction is single-drug, pair, or combination.", space_after=10)
    add_bullet_item(tf4_l, "Intuitive Evidence Ladders",
                    "Provides visual evidence charts displaying exact risk estimates across sub-groups so clinical teams can evaluate findings at a glance.", space_after=10)
    add_bullet_item(tf4_l, "Extreme Performance & Cost Advantage",
                    "Columnar local querying via DuckDB delivers sub-second analysis over millions of records, eliminating costly enterprise cloud infrastructure.", space_after=10)
    add_bullet_item(tf4_l, "Auditable Scientific Transparency",
                    "Maintains complete transparency by separating everyday plain-language pages from a full mathematical and exclusion glossary.", space_after=10)

    tf4_r = add_card(s4, Inches(6.8), top4, Inches(5.75), Inches(5.35), "Startup Wedge & Revenue Model")
    add_bullet_item(tf4_r, "Freemium Researcher Wedge",
                    "Offers 100 free exploratory queries per day to capture medical residents, university researchers, and faculty, building bottom-up adoption.", space_after=10)
    add_bullet_item(tf4_r, "Institutional SaaS Tier",
                    "Provides unlimited searches, team collaboration, saved drug-cocktail alerts, and custom hospital exclusion lists at an affordable institutional rate.", space_after=10)
    add_bullet_item(tf4_r, "Hospital IT & EHR API Integration",
                    "Programmatic REST API tier allowing electronic health record vendors and pharmacy dispensing systems to query combination risks in real time.", space_after=10)
    add_bullet_item(tf4_r, "Automated Regulatory Audit Exports",
                    "Generates one-click PDF safety audit packages formatted for pharmacovigilance compliance reviews and regulatory submissions.", space_after=10)
    add_bullet_item(tf4_r, "Competitive Wedge Against Legacy Tools",
                    "OpenVigil lacks polypharmacy and modern UI; enterprise suites cost hundreds of thousands. PharmaSense captures the underserved institutional middle.", space_after=10)

    set_speaker_notes(s4,
        "From a startup perspective, PharmaSense addresses the large unserved middle market between zero-insight free dashboards "
        "and six-figure enterprise suites. Our product delivers plain-language clinical verdicts rather than raw math, allowing pharmacists "
        "to act immediately. We plan a freemium wedge for academic researchers coupled with affordable institutional subscriptions and "
        "an embeddable API for pharmacy software systems."
    )

    # --------------------------------------------------------------------------
    # SLIDE 5: Data Sources and Preparation
    # --------------------------------------------------------------------------
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5, LIGHT_BG)
    top5 = add_header(s5, "04 / Data Sources & Ingestion Engineering",
                      "Extracting Clean Polypharmacy Cohorts from openFDA",
                      "Robust data harvesting, authentication, and ingredient-level entity resolution")

    tf5_l = add_card(s5, Inches(0.75), top5, Inches(5.75), Inches(5.35), "API Harvesting & Pipeline Engineering")
    add_bullet_item(tf5_l, "Direct openFDA REST Integration",
                    "Connects directly to the official openFDA drug adverse event API endpoint, fetching reports containing 3 or more concurrent medications.", space_after=10)
    add_bullet_item(tf5_l, "Pre-Flight Connectivity & Key Validation",
                    "Executes an immediate lightweight test call to verify API key validity and network availability before initiating large downloads, failing fast if needed.", space_after=10)
    add_bullet_item(tf5_l, "Paging & Month-Wise Stratification",
                    "Overcomes API pagination limits by dividing queries across all 12 calendar months, preventing sampling bias caused by front-loaded annual records.", space_after=10)
    add_bullet_item(tf5_l, "Robust Network Resilience",
                    "Implements exponential backoff and automatic retry logic across rate-limiting errors and server timeouts, ensuring uninterrupted data extraction.", space_after=10)
    add_bullet_item(tf5_l, "Case De-Duplication by Manufacturer ID",
                    "Merges multiple follow-up versions of the same case by grouping on unique manufacturer case numbers, preserving only the final consolidated record.", space_after=10)

    tf5_r = add_card(s5, Inches(6.8), top5, Inches(5.75), Inches(5.35), "Entity Resolution & Metadata Preservation")
    add_bullet_item(tf5_r, "Preservation of Critical Regulatory Fields",
                    "Retains regulatory drug roles (suspect vs concomitant), patient recorded indications, generic ingredient names, and reported adverse reactions.", space_after=10)
    add_bullet_item(tf5_r, "Ingredient-Level Normalization Pipeline",
                    "Uses regex tokenizers to strip dosage quantities (e.g. 500 MG), physical delivery forms (TABLET, INJECTION), and chemical salts (HYDROCHLORIDE).", space_after=10)
    add_bullet_item(tf5_r, "Brand-to-Generic Alias Merging",
                    "Maps commercial trade names to active generic ingredients, ensuring that 'Glucophage' and 'Metformin' are counted as one ingredient rather than a combination.", space_after=10)
    add_bullet_item(tf5_r, "Elimination of Administrative Placeholders",
                    "Purges useless database tokens like 'UNKNOWN DRUG', 'UNSPECIFIED', or 'PRODUCT' that would otherwise create spurious drug combinations.", space_after=10)
    add_bullet_item(tf5_r, "Columnar Parquet Serialization",
                    "Stores harvested records in compressed Parquet format, enabling rapid reload and iteration without repeated openFDA network requests.", space_after=10)

    set_speaker_notes(s5,
        "Real-world FDA data is exceptionally messy. Our ingestion engine stratifies requests across all 12 months to avoid "
        "temporal bias, validates authentication upfront, and groups cases by manufacturer IDs to eliminate duplicates. "
        "More importantly, our entity resolution pipeline strips dosages, delivery forms, and salts, unifying brand names "
        "and generics into true active ingredients while removing meaningless placeholder terms."
    )

    # --------------------------------------------------------------------------
    # SLIDE 6: Data Cleaning and Quality Decisions
    # --------------------------------------------------------------------------
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6, LIGHT_BG)
    top6 = add_header(s6, "05 / Clinical Cleaning & Quality Decisions",
                      "Eliminating False Inflation and Confounding at the Source",
                      "Targeted heuristic rules that prevent spurious signals before any statistics are calculated")

    tf6_l = add_card(s6, Inches(0.75), top6, Inches(5.75), Inches(5.35), "Targeted Clinical Cleaning Decisions")
    add_bullet_item(tf6_l, "Removal of Indication Confounding",
                    "Removes any reported reaction term that matches the recorded indication of a drug in the same report (e.g., treating hypertension vs causing it).", space_after=10)
    add_bullet_item(tf6_l, "Why Indication Removal is Mandatory",
                    "Counting pre-existing diseases as adverse events causes risk ratios to explode into the hundreds because unexposed patients rarely have that disease.", space_after=10)
    add_bullet_item(tf6_l, "Curated Non-Adverse Event Exclusion List",
                    "Maintains an editable dictionary of ~70 non-event terms across administrative categories, including off-label use, drug ineffectiveness, and paperwork terms.", space_after=10)
    add_bullet_item(tf6_l, "Eliminating Misleading Vague Terms",
                    "Terms like 'Feeling abnormal' or product quality complaints are heavily clustered and distort mathematical baseline comparisons across patient cohorts.", space_after=10)
    add_bullet_item(tf6_l, "Indication-Reaction Overlap Removal",
                    "Ensures that true adverse reactions are separated from underlying medical conditions being treated by the drug regimen.", space_after=10)

    tf6_r = add_card(s6, Inches(6.8), top6, Inches(5.75), Inches(5.35), "Data Quality Choices & Cohort Integrity")
    add_bullet_item(tf6_r, "Rejection of openFDA Duplicate Flag",
                    "We discovered that openFDA's internal 'duplicate=1' flag is incorrectly assigned to almost all records; filtering on it would erase the entire dataset.", space_after=10)
    add_bullet_item(tf6_r, "Requirement for 3 Distinct Active Ingredients",
                    "Enforces that every retained report must contain at least 3 distinct pharmacological active ingredients after all entity merging and alias resolution.", space_after=10)
    add_bullet_item(tf6_r, "Genuine Adverse Event Availability Filter",
                    "Drops reports where all listed reactions were non-event administrative terms or recorded indications, leaving only clinically meaningful reports.", space_after=10)
    add_bullet_item(tf6_r, "Transparent Exclusion Logging",
                    "Logs every dropped reaction term and occurrence frequency into an audit table viewable on the dashboard for regulatory verification.", space_after=10)
    add_bullet_item(tf6_r, "Measurable Yield from Cleaning",
                    "Out of 2,998 raw reports, our cleaning preserved 2,008 high-quality cases, dropping 758 with fewer than 3 ingredients and 232 with no genuine reaction.", space_after=10)

    set_speaker_notes(s6,
        "Two critical clinical decisions define our cleaning pipeline. First, we remove any reaction that matches the patient's "
        "own drug indication—a drug prescribed for diabetes shouldn't be falsely accused of causing diabetes. Second, we filter out "
        "around 70 non-medical administrative terms like 'Off Label Use' and 'Drug Ineffective' that artificially inflate risk ratios. "
        "We also intentionally avoided openFDA's flawed duplicate flag, which would have wiped out valid records."
    )

    # --------------------------------------------------------------------------
    # SLIDE 7: End-to-End Pipeline Diagram (Text-heavy & fully populated)
    # --------------------------------------------------------------------------
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7, LIGHT_BG)
    top7 = add_header(s7, "06 / End-to-End Pipeline Architecture",
                      "Modular Data Flow from openFDA to Clinical Dashboard",
                      "Visual schematic of the 8 stages powering signal discovery, verification, and presentation")

    steps = [
        ("1. Fetch & Ingest", "openFDA REST API",
         "Harvests polypharmacy reports (>=3 drugs) with month-wise pagination.",
         "Validates API keys upfront with exponential backoff on rate limits."),
        ("2. Clean & Normalise", "Entity Resolution",
         "Unifies brand names to active ingredients; strips dosages, forms, and salts.",
         "Excludes disease indications and ~70 non-event paperwork terms."),
        ("3. Load to DuckDB", "High-Speed OLAP",
         "Converts clean reports into relational demographics, drug, and reaction tables.",
         "Enables lightning-fast local analytical SQL queries without server overhead."),
        ("4. Mine Combinations", "FP-Growth Algorithm",
         "Discovers candidate 3-drug triplets meeting minimum patient support limits.",
         "Restricts itemset mining to the 300 most frequent active ingredients."),
        ("5. Multi-Gate Detection", "Disproportionality",
         "Evaluates candidate triplets across 8 mutually exclusive patient cohorts.",
         "Applies exact significance tests and false-discovery rate corrections."),
        ("6. Attribution Engine", "Clinical Decision Tree",
         "Compares 3-drug risk against all single drugs and pairs head-to-head.",
         "Categorizes signals into single-drug, pair, or combination verdicts."),
        ("7. Machine Learning", "Predictive Modeling",
         "Trains Lasso L1, Random Forest, and XGBoost classifiers for outcome risk.",
         "Evaluates accuracy, recall, and AUPRC while ranking top risk-factor drugs."),
        ("8. Streamlit Dashboard", "Interactive Web UI",
         "Renders plain-language search, evidence ladders, and formula glossary.",
         "Provides instant interactive filtering without technical or math barriers.")
    ]

    card_w = Inches(2.78)
    card_h = Inches(2.55)
    gap_x = Inches(0.22)
    start_x = Inches(0.75)
    row1_y = top7 + Inches(0.08)
    row2_y = top7 + Inches(2.75)

    for i, (stitle, ssub, sline1, sline2) in enumerate(steps):
        r = i // 4
        c = i % 4
        x = start_x + c * (card_w + gap_x)
        y = row1_y if r == 0 else row2_y
        
        bg_col = CARD_ALT if (i % 2 == 1) else CARD_BG
        tf_step = add_card(s7, x, y, card_w, card_h, stitle, bg_color=bg_col)
        
        p_sub = tf_step.add_paragraph()
        p_sub.text = ssub.upper()
        p_sub.font.name = FONT_HEADING
        p_sub.font.size = Pt(9.2)
        p_sub.font.bold = True
        p_sub.font.color.rgb = TEAL
        p_sub.space_after = Pt(6)

        add_bullet_item(tf_step, "Mechanism", sline1, font_size=9.2, space_after=5)
        add_bullet_item(tf_step, "Benefit", sline2, font_size=9.2, space_after=4)

    set_speaker_notes(s7,
        "This flow diagram illustrates our 8-stage pipeline. We ingest raw reports, clean and normalize them to ingredients, "
        "and load them into a local columnar DuckDB database. We then run FP-Growth to discover candidate drug triplets, "
        "pass them through our multi-gate disproportionality tests, apply our attribution engine to assign plain-language verdicts, "
        "train predictive ML models, and serve everything through a fast Streamlit interface."
    )

    # --------------------------------------------------------------------------
    # SLIDE 8: Multi-Gate Detection Logic (in words)
    # --------------------------------------------------------------------------
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8, LIGHT_BG)
    top8 = add_header(s8, "07 / Multi-Gate Detection Architecture",
                      "How PharmaSense Filters Thousands of False Alarms",
                      "Six sequential statistical safeguards described in plain language without mathematical formulas")

    tf8_l = add_card(s8, Inches(0.75), top8, Inches(5.75), Inches(5.35), "The First Three Gates: Clinical Relevance")
    add_bullet_item(tf8_l, "Gate 1: Minimum Patient Support Filter",
                    "Requires a 3-drug combination to appear in at least 10 independent patient cases, and the specific reaction to occur in at least 3 cases, eliminating random one-off events.", space_after=10)
    add_bullet_item(tf8_l, "Gate 2: Pure Reference Disproportionality",
                    "Compares reporting frequency of the reaction when all three drugs are present against the pure baseline cohort where none of the three drugs were taken.", space_after=10)
    add_bullet_item(tf8_l, "Why the Pure Baseline is Critical",
                    "Comparing against 'everyone else' allows popular unrelated drugs to mask small combination signals; comparing to an unexposed cohort ensures clean, unpolluted contrast.", space_after=10)
    add_bullet_item(tf8_l, "Gate 3: Confidence Lower-Bound Verification",
                    "Requires that the conservative lower bound of the estimated risk ratio remains strictly above 1, ensuring elevated reporting odds with statistical confidence.", space_after=10)
    add_bullet_item(tf8_l, "Preventing Small-Sample False Alarms",
                    "If the lower confidence bound drops below 1, the signal is rejected, guaranteeing that wide intervals from small numbers cannot produce false alarms.", space_after=10)

    tf8_r = add_card(s8, Inches(6.8), top8, Inches(5.75), Inches(5.35), "The Final Three Gates: Rigor & Attribution")
    add_bullet_item(tf8_r, "Gate 4: Exact Significance & Multiple Testing",
                    "Calculates an exact probability test conditioned on the minimum-event threshold, followed by false discovery rate correction across all thousands of simultaneous tests.", space_after=10)
    add_bullet_item(tf8_r, "Why Standard Testing Fails Here",
                    "Standard large-sample tests become dangerously over-optimistic when adverse event counts are small; exact testing with false discovery correction prevents false leads.", space_after=10)
    add_bullet_item(tf8_r, "Gate 5: Unstable Estimate & Rarity Flagging",
                    "Flags any pattern where the reaction almost never occurs without the drugs, alerting clinicians when a mathematical ratio is volatile due to baseline rarity.", space_after=10)
    add_bullet_item(tf8_r, "Gate 6: Head-to-Head Sub-Cohort Superiority",
                    "Tests whether the 3-drug cohort has statistically higher reaction odds than each of its 6 sub-groups (the 3 single drugs and 3 dual pairs taken alone).", space_after=10)
    add_bullet_item(tf8_r, "The Final Verdict Assignment",
                    "Only combinations that successfully clear all gates and beat all sub-groups receive the formal verdict of 'Combination effect'.", space_after=10)

    set_speaker_notes(s8,
        "To ensure clinicians can trust our findings, we built a 6-gate detection architecture. First, we enforce minimum case counts "
        "and compare the triplet against a pure unexposed baseline. We verify that the conservative lower confidence bound exceeds 1. "
        "Then, we run exact significance testing with multiple-testing correction across all comparisons, flag unstable estimates, "
        "and test the 3-drug group head-to-head against all single drugs and pairs to confirm true combination synergy."
    )

    # --------------------------------------------------------------------------
    # SLIDE 9: Attribution Engine & Clinical Scenarios
    # --------------------------------------------------------------------------
    s9 = prs.slides.add_slide(blank_layout)
    set_slide_background(s9, LIGHT_BG)
    top9 = add_header(s9, "08 / Attribution Logic & Proven Scenarios",
                      "Resolving Blame Across Single Drugs, Pairs, and Triplets",
                      "How the decision tree separates true synergy from innocent bystander co-medications")

    tf9_l = add_card(s9, Inches(0.75), top9, Inches(5.75), Inches(5.35), "Attribution Hierarchy & Categorization")
    add_bullet_item(tf9_l, "Single-Drug Effect (The Innocent Bystanders)",
                    "Assigned when one drug on its own fully explains the elevated risk, and adding companion medications provides no significant risk increase; the others are bystanders.", space_after=10)
    add_bullet_item(tf9_l, "Pair Effect (Two-Drug Interaction)",
                    "Assigned when two drugs together explain the elevated signal, but adding the third drug does not increase risk, identifying the third medication as an innocent passenger.", space_after=10)
    add_bullet_item(tf9_l, "Combination Effect (True 3-Way Synergy)",
                    "Assigned when the 3-drug group shows statistically higher odds than all single drugs and all pairs, surviving strict multiple-testing correction.", space_after=10)
    add_bullet_item(tf9_l, "Probable Combination (Sub-Group Sparsity)",
                    "Assigned when the 3-drug group exceeds all sub-groups, but some sub-groups have too few reports to completely rule out a single-drug cause.", space_after=10)
    add_bullet_item(tf9_l, "Cannot Separate (Tight Co-Prescription)",
                    "Assigned when the medications are almost always prescribed together, making it statistically impossible to isolate individual drug responsibility.", space_after=10)

    tf9_r = add_card(s9, Inches(6.8), top9, Inches(5.75), Inches(5.35), "Validation Across 5 Real-World Scenarios")
    add_bullet_item(tf9_r, "Scenario 1: Digoxin + Quinidine Pair",
                    "Accurately categorized as a 'Pair effect' driven by Digoxin toxicity exacerbated by Quinidine, successfully avoiding false blame on third co-medications.", space_after=10)
    add_bullet_item(tf9_r, "Scenario 2: Levofloxacin Alone",
                    "Correctly isolated as a 'Single-drug effect' for tendonitis, recognizing that companion medications were innocent bystanders.", space_after=10)
    add_bullet_item(tf9_r, "Scenario 3: Prednisone with Levofloxacin",
                    "The system clearly justified: 'Reaction tracks Levofloxacin, not Prednisone', preventing false attribution to the co-prescribed steroid.", space_after=10)
    add_bullet_item(tf9_r, "Scenario 4: Metformin Alone",
                    "Correctly flagged lactic acidosis as driven by Metformin alone, dismissing companion blood pressure medications from responsibility.", space_after=10)
    add_bullet_item(tf9_r, "Scenario 5: Warfarin + Amiodarone + Simvastatin",
                    "Successfully flagged as a true 'Combination effect' for muscle breakdown (Rhabdomyolysis), proving the engine captures genuine 3-way interactions.", space_after=10)

    set_speaker_notes(s9,
        "Our attribution logic solves the fundamental clinical question: which drug is actually responsible? "
        "By comparing the 3-drug group against its 6 sub-groups, our decision tree isolates single-drug effects, pair effects, "
        "and true combination synergy. We tested this against five established pharmacological ground-truth scenarios, "
        "and in all five cases—including Digoxin-Quinidine pair toxicity and the Warfarin-Amiodarone-Simvastatin triplet—the "
        "engine assigned the exact right verdict."
    )

    # --------------------------------------------------------------------------
    # SLIDE 10: The Product (Dashboard Walkthrough)
    # --------------------------------------------------------------------------
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10, LIGHT_BG)
    top10 = add_header(s10, "09 / The Product: Clinical Dashboard",
                      "Accessible, Plain-Language Safety Interface",
                      "Designed for clinical workflows: intuitive exploration with complete methodological transparency")

    tf10_l = add_card(s10, Inches(0.75), top10, Inches(5.75), Inches(5.35), "Core Clinical Pages & Workflows")
    add_bullet_item(tf10_l, "Overview Analytics Dashboard",
                    "Displays cohort metrics, distribution of mined reports, and risk histograms showing that most background co-prescriptions sit near baseline risk.", space_after=10)
    add_bullet_item(tf10_l, "Single-Page Drug Search Workflow",
                    "Enables a clinician to select a drug, inspect its reported adverse events, and immediately determine whether an event is caused by the drug alone or partner drugs.", space_after=10)
    add_bullet_item(tf10_l, "Combination Signals Explorer",
                    "Presents all mined 3-drug safety signals with dynamic filters for case counts, clinical verdicts, and specific adverse event categories.", space_after=10)
    add_bullet_item(tf10_l, "Intuitive Evidence Ladder Charts",
                    "Visualizes comparative reporting odds across all sub-groups using forest plots with confidence bars, accompanied by plain-language clinical justifications.", space_after=10)
    add_bullet_item(tf10_l, "Partner Drug Setting-Aside Feature",
                    "Automatically sets aside the dominant partner medication and re-tests remaining reports, confirming whether a drug is a true driver or a fellow traveler.", space_after=10)

    tf10_r = add_card(s10, Inches(6.8), top10, Inches(5.75), Inches(5.35), "Transparency & Advanced Intelligence")
    add_bullet_item(tf10_r, "Dedicated Glossary & Formulas Page",
                    "Completely separates clinical workflows from mathematical details; everyday users see plain words, while epidemiologists can audit all formulas on this page.", space_after=10)
    add_bullet_item(tf10_r, "Exclusion Catalog Inspection",
                    "Allows users to review all ~70 excluded terms, inspect their removal frequencies, and verify why specific terms were classified as non-events.", space_after=10)
    add_bullet_item(tf10_r, "Machine Learning Insights Page",
                    "Displays model evaluation benchmarks (Accuracy, Precision, Recall, ROC-AUC) and plots feature importance identifying top drug risk predictors.", space_after=10)
    add_bullet_item(tf10_r, "Sub-Second Local Execution",
                    "Engineered with DuckDB and Streamlit caching to respond to user filter changes in milliseconds, ensuring a seamless user experience.", space_after=10)
    add_bullet_item(tf10_r, "Designed for Clinical Trust",
                    "Avoids opaque black-box scores by pairing every statistical decision with clear evidentiary justification and sample-size context.", space_after=10)

    set_speaker_notes(s10,
        "The PharmaSense dashboard is designed specifically for healthcare practitioners who need clarity without mathematical friction. "
        "The Drug Search page lets users pick a medication and immediately inspect partner drug attributions and visual evidence ladders. "
        "To maintain usability, all clinical pages use plain English, while all underlying formulas, decision trees, and exclusion audits "
        "are housed on a dedicated Glossary page for complete transparency."
    )

    # --------------------------------------------------------------------------
    # SLIDE 11: What We Have Achieved So Far
    # --------------------------------------------------------------------------
    s11 = prs.slides.add_slide(blank_layout)
    set_slide_background(s11, LIGHT_BG)
    top11 = add_header(s11, "10 / Current Achievements & Milestone Audit",
                      "Measurable Technical Progress Delivered in Checkpoint 2",
                      "Honest assessment of delivered components, algorithmic fixes, and validation benchmarks")

    tf11_l = add_card(s11, Inches(0.75), top11, Inches(5.75), Inches(5.35), "Delivered Components & Engine Stability")
    add_bullet_item(tf11_l, "✓ Fully Runnable Local Architecture",
                    "Executed and verified the entire pipeline from ingestion through database creation to the interactive web dashboard on localhost.", space_after=10)
    add_bullet_item(tf11_l, "✓ Resolved Risk Ratio Inflation",
                    "Diagnosed and resolved the severe statistical inflation issue by enforcing ingredient normalization, indication filtering, and pure baseline comparison.", space_after=10)
    add_bullet_item(tf11_l, "✓ Robust DuckDB SQL Optimization",
                    "Replaced fragile, slow query loops with vectorized DuckDB tables and sparse matrix cross-products, eliminating crashes on drug names with apostrophes.", space_after=10)
    add_bullet_item(tf11_l, "✓ Integrated Quality Exclusion Module",
                    "Developed the modular exclusions architecture to systematically strip administrative and non-adverse event noise from raw reports.", space_after=10)
    add_bullet_item(tf11_l, "✓ Machine Learning Benchmarking",
                    "Trained and evaluated L1 Lasso, Random Forest, and XGBoost baseline classifiers, achieving >0.96 ROC-AUC on adverse event target classification.", space_after=10)

    tf11_r = add_card(s11, Inches(6.8), top11, Inches(5.75), Inches(5.35), "Testing Benchmarks & Honest Validation Status")
    add_bullet_item(tf11_r, "✓ Automated Self-Checking Test Suite",
                    "Built an automated test suite verifying mathematical consistency, zero-cell corrections, and attribution logic in seconds before every run.", space_after=10)
    add_bullet_item(tf11_r, "✓ Ground-Truth Recovery on Planted Data",
                    "Successfully recovered and validated planted combination interactions and single-drug controls with only 1 false alarm out of 60 random pairs.", space_after=10)
    add_bullet_item(tf11_r, "✓ Comprehensive Code Documentation",
                    "Documented full codebase changes, setup steps, architectural decisions, and usage commands in dedicated README and migration logs.", space_after=10)
    add_bullet_item(tf11_r, "⚠️ Transparent Validation Status",
                    "Testing thus far has been rigorously conducted on synthetic openFDA-structured cohorts; conducting large-scale live API validation is our immediate next step.", space_after=10)
    add_bullet_item(tf11_r, "Readiness for Live Deployment",
                    "The codebase is fully equipped with authentication, rate-limiting handlers, and error recovery, ready for live-data harvesting in Phase 3.", space_after=10)

    set_speaker_notes(s11,
        "Here is our progress checklist for Checkpoint 2. We have achieved a fully working, reproducible local pipeline. "
        "We successfully fixed critical mathematical bugs that were causing ratio inflation, optimized SQL aggregation in DuckDB, "
        "and built an automated test harness that passes in seconds. To be completely transparent with the committee: "
        "our validation so far has used synthetic openFDA-shaped cohorts with planted ground truth, and live API runs are scheduled next."
    )

    # --------------------------------------------------------------------------
    # SLIDE 12: Challenges and Limitations
    # --------------------------------------------------------------------------
    s12 = prs.slides.add_slide(blank_layout)
    set_slide_background(s12, LIGHT_BG)
    top12 = add_header(s12, "11 / Methodological Challenges & Limitations",
                      "Navigating the Inherent Constraints of Spontaneous Reports",
                      "Scientific hurdles encountered in pharmacovigilance and our strategies to address them")

    tf12_l = add_card(s12, Inches(0.75), top12, Inches(5.75), Inches(5.35), "Statistical & Epidemiological Hurdles")
    add_bullet_item(tf12_l, "Confounding by Unseen Fourth Drugs",
                    "Attribution analyzes the three drugs in each combination; if a patient takes an unanalyzed fourth drug that is the real culprit, it can distort attribution.", space_after=10)
    add_bullet_item(tf12_l, "Concomitant vs Suspect Drug Inclusion",
                    "By default, all listed drugs are analyzed to capture overlooked interactions; however, including concomitant medications introduces background noise.", space_after=10)
    add_bullet_item(tf12_l, "Sparsity of True 3-Drug Combinations",
                    "In real clinical data, true synergistic 3-way interactions are biologically rare; we anticipate only a small percentage of combinations will show true synergy.", space_after=10)
    add_bullet_item(tf12_l, "Underpowered Interaction Regressions",
                    "Standard statistical regressions frequently fail to converge when specific drug-event combination cells contain very few patient reports.", space_after=10)
    add_bullet_item(tf12_l, "Mitigation via Partner Setting-Aside",
                    "We mitigate confounding by automatically setting aside the strongest partner medication and re-testing to ensure companion drugs are not falsely blamed.", space_after=10)

    tf12_r = add_card(s12, Inches(6.8), top12, Inches(5.75), Inches(5.35), "Data Source & Regulatory Constraints")
    add_bullet_item(tf12_r, "Spontaneous Reporting Biases",
                    "FAERS represents voluntary reporting rather than controlled incidence; reports are heavily influenced by media attention, litigation, and drug age.", space_after=10)
    add_bullet_item(tf12_r, "Lack of a Licensed MedDRA Dictionary",
                    "Standard clinical trials map terms to hierarchical MedDRA classes; because MedDRA requires commercial licensing, we rely on curated heuristic exclusions.", space_after=10)
    add_bullet_item(tf12_r, "Association Does Not Equal Causation",
                    "Spontaneous statistical signals identify statistical disproportionality only; they provide early research warnings but cannot prove biological causality.", space_after=10)
    add_bullet_item(tf12_r, "Sample Truncation Artifacts",
                    "openFDA returns records chronologically rather than a true random sample of FAERS, necessitating month-by-month slicing to prevent seasonal bias.", space_after=10)
    add_bullet_item(tf12_r, "Ethical AI Safety Constraints",
                    "Every finding in the product is explicitly flagged as a research signal, ensuring that algorithmic outputs are never treated as direct clinical advice.", space_after=10)

    set_speaker_notes(s12,
        "Every scientific project must acknowledge its boundaries. Spontaneous reporting in FAERS is subject to reporting bias and "
        "confounding from unobserved fourth medications. True 3-drug synergies are biologically rare, meaning we expect few positive "
        "combination verdicts on real data. Additionally, without a proprietary MedDRA license, we rely on curated exclusion lists. "
        "We address confounding directly by setting aside dominant partner drugs and re-testing to avoid blaming bystanders."
    )

    # --------------------------------------------------------------------------
    # SLIDE 13: Next Steps and Roadmap (Timeline Graphic)
    # --------------------------------------------------------------------------
    s13 = prs.slides.add_slide(blank_layout)
    set_slide_background(s13, LIGHT_BG)
    top13 = add_header(s13, "12 / Strategic Roadmap to Launch",
                      "Phase-by-Phase Plan for Checkpoint 3 and Product Launch",
                      "Clear timeline spanning live validation, regulatory label cross-referencing, and deployment")

    phases = [
        ("Phase 2b · Late October", "Live Data Validation & Audit",
         [("Full Live Harvesting", "Execute multi-month live openFDA harvest (20,000+ real-world reports)."),
          ("Clinical Expert Review", "Review discovered signals with pharmacy professors and clinical toxicologists."),
          ("Threshold Optimization", "Tune minimum support and event thresholds based on real data yield."),
          ("Scale Stress Testing", "Stress-test pipeline runtime across expanded multi-gigabyte cohorts."),
          ("Signal Cataloguing", "Establish a verified benchmark catalog of known vs suspicious signals.")]),
        ("Phase 3a · Early November", "FDA Drug Label Integration",
         [("Drug Label Scraping", "Scrape official openFDA Drug Label API (140,000+ package inserts)."),
          ("Label Cross-Referencing", "Automatically compare mined signals against known black-box warnings."),
          ("Signal Categorization", "Classify signals into 'Known / Documented' versus 'Novel Emergent'."),
          ("Novel Hazard Highlighting", "Highlight unreported drug combinations as prime targets for researchers."),
          ("Discrepancy Auditing", "Quantify how many real-world adverse events are completely absent from labels.")]),
        ("Phase 3b · Mid November", "Emerging Signal Scoring & Trends",
         [("Temporal Velocity Tracking", "Measure month-over-month growth rates in adverse event filings."),
          ("Emerging Signal Score", "Construct composite metric combining growth velocity, severity, and odds."),
          ("Urgent Signal Ranking", "Rank rapidly accelerating combinations for immediate regulatory review."),
          ("Confounding Adjustment", "Strengthen confounding control using multivariate propensity scoring."),
          ("Interactive Trend Plots", "Embed interactive timeline charts showing signal emergence over months.")]),
        ("Phase 4 · End November", "Productization & Cloud Launch",
         [("Docker Containerization", "Package the application into reproducible Docker containers."),
          ("Live Cloud Deployment", "Deploy the web platform on cloud infrastructure (Render / Streamlit Cloud)."),
          ("One-Click Safety Audits", "Implement automated PDF and CSV safety audit report generation."),
          ("FastAPI Microservices", "Release REST API endpoints for third-party hospital IT integration."),
          ("User Pilot Feedback", "Collect user experience feedback from hospital and academic pharmacists.")])
    ]

    phase_w = Inches(2.78)
    phase_h = Inches(5.35)
    phase_gap = Inches(0.22)

    for i, (p_title, p_sub, p_items) in enumerate(phases):
        px = start_x + i * (phase_w + phase_gap)
        tf_phase = add_card(s13, px, top13, phase_w, phase_h, p_title,
                            bg_color=CARD_BG if i % 2 == 0 else CARD_ALT)
        
        p_subt = tf_phase.add_paragraph()
        p_subt.text = p_sub.upper()
        p_subt.font.name = FONT_HEADING
        p_subt.font.size = Pt(9.5)
        p_subt.font.bold = True
        p_subt.font.color.rgb = TEAL
        p_subt.space_after = Pt(8)

        for head, body in p_items:
            add_bullet_item(tf_phase, head, body, font_size=9.2, space_after=6)

    set_speaker_notes(s13,
        "Our roadmap to the final evaluation follows a structured four-phase path. In late October, we conduct full live openFDA runs "
        "and review signals. In early November, we integrate the openFDA Drug Label API to automatically distinguish novel combinations "
        "from already-documented side effects. In mid-November, we implement temporal velocity tracking to rank emerging signals, "
        "and by late November we will deploy the containerized platform online with automated PDF report exports."
    )

    # --------------------------------------------------------------------------
    # SLIDE 14: Conclusion & Impact Summary (High contrast & perfectly filled)
    # --------------------------------------------------------------------------
    s14 = prs.slides.add_slide(blank_layout)
    set_slide_background(s14, DARK_BG)
    
    acc14 = s14.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.28), Inches(7.5))
    acc14.fill.solid()
    acc14.fill.fore_color.rgb = MINT
    acc14.line.fill.background()

    tb14_tag = s14.shapes.add_textbox(Inches(1.0), Inches(0.48), Inches(11.0), Inches(0.35))
    tf14_tag = tb14_tag.text_frame
    p14_tag = tf14_tag.paragraphs[0]
    p14_tag.text = "SUMMARY & CONCLUSION · CHECKPOINT 2 EVALUATION"
    p14_tag.font.name = FONT_HEADING
    p14_tag.font.size = Pt(11)
    p14_tag.font.bold = True
    p14_tag.font.color.rgb = MINT

    tb14_t = s14.shapes.add_textbox(Inches(1.0), Inches(0.85), Inches(11.0), Inches(1.1))
    tf14_t = tb14_t.text_frame
    p14_t = tf14_t.paragraphs[0]
    p14_t.text = "Transforming Real-World Data into Searchable Safety Intelligence"
    p14_t.font.name = FONT_HEADING
    p14_t.font.size = Pt(26)
    p14_t.font.bold = True
    p14_t.font.color.rgb = WHITE

    # Two Main Summary Cards with HIGH CONTRAST (White text on dark cards)
    card14_l = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(2.05), Inches(5.35), Inches(3.9))
    card14_l.fill.solid()
    card14_l.fill.fore_color.rgb = DARK_CARD_BG
    card14_l.line.color.rgb = TEAL
    card14_l.line.width = Pt(1.5)

    tb14_l = s14.shapes.add_textbox(Inches(1.24), Inches(2.22), Inches(4.87), Inches(3.6))
    tf14_l = tb14_l.text_frame
    tf14_l.word_wrap = True
    
    p14_lt = tf14_l.paragraphs[0]
    p14_lt.text = "Key Achievements in Checkpoint 2"
    p14_lt.font.name = FONT_HEADING
    p14_lt.font.size = Pt(13)
    p14_lt.font.bold = True
    p14_lt.font.color.rgb = MINT
    p14_lt.space_after = Pt(8)

    add_bullet_item(tf14_l, "Proven Multi-Gate Attribution",
                    "Successfully proved that complex 3-drug polypharmacy risk can be systematically decomposed into clear, plain-language clinical verdicts with exact statistics.",
                    font_size=10.2, space_after=7, head_color=WHITE, body_color=TEXT_LIGHT)
    add_bullet_item(tf14_l, "Eliminated False Signal Inflation",
                    "Resolved historical ratio inflation through strict entity normalization, non-event filtering, and pure baseline comparisons.",
                    font_size=10.2, space_after=7, head_color=WHITE, body_color=TEXT_LIGHT)
    add_bullet_item(tf14_l, "High-Speed Columnar Architecture",
                    "DuckDB backend and sparse matrix multiplication deliver sub-second queries across millions of patient records.",
                    font_size=10.2, space_after=7, head_color=WHITE, body_color=TEXT_LIGHT)
    add_bullet_item(tf14_l, "Fully Functional Local Web Application",
                    "The backend mining, DuckDB database, predictive ML models, and Streamlit dashboard are fully integrated and running on localhost.",
                    font_size=10.2, space_after=7, head_color=WHITE, body_color=TEXT_LIGHT)

    card14_r = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.6), Inches(2.05), Inches(5.35), Inches(3.9))
    card14_r.fill.solid()
    card14_r.fill.fore_color.rgb = DARK_CARD_BG
    card14_r.line.color.rgb = TEAL
    card14_r.line.width = Pt(1.5)

    tb14_r = s14.shapes.add_textbox(Inches(6.84), Inches(2.22), Inches(4.87), Inches(3.6))
    tf14_r = tb14_r.text_frame
    tf14_r.word_wrap = True
    
    p14_rt = tf14_r.paragraphs[0]
    p14_rt.text = "Clinical Impact & Startup Potential"
    p14_rt.font.name = FONT_HEADING
    p14_rt.font.size = Pt(13)
    p14_rt.font.bold = True
    p14_rt.font.color.rgb = MINT
    p14_rt.space_after = Pt(8)

    add_bullet_item(tf14_r, "Democratizing Drug Safety Intelligence",
                    "PharmaSense provides hospitals and researchers with a powerful analytical alternative to unaffordable enterprise pharmacovigilance suites.",
                    font_size=10.2, space_after=7, head_color=WHITE, body_color=TEXT_LIGHT)
    add_bullet_item(tf14_r, "Accelerating Clinical Decisions",
                    "Empowers clinical pharmacists to rapidly discern whether a patient's adverse reaction is caused by an emerging interaction or a single drug.",
                    font_size=10.2, space_after=7, head_color=WHITE, body_color=TEXT_LIGHT)
    add_bullet_item(tf14_r, "Commercial SaaS Trajectory",
                    "Structured for immediate commercialization with freemium researcher tiers, institutional subscriptions, and embeddable hospital APIs.",
                    font_size=10.2, space_after=7, head_color=WHITE, body_color=TEXT_LIGHT)
    add_bullet_item(tf14_r, "Validated Ethical AI Guardrails",
                    "Embeds clear ethical constraints: statistical association is not biological causation; all outputs are structured for research guidance.",
                    font_size=10.2, space_after=7, head_color=WHITE, body_color=TEXT_LIGHT)

    # Thank you & Contact Footer Box
    b14_foot = s14.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(6.15), Inches(11.0), Inches(1.0))
    b14_foot.fill.solid()
    b14_foot.fill.fore_color.rgb = RGBColor(0x0E, 0x22, 0x29)
    b14_foot.line.color.rgb = MINT
    b14_foot.line.width = Pt(1.2)

    tb14_foot = s14.shapes.add_textbox(Inches(1.24), Inches(6.22), Inches(10.5), Inches(0.85))
    tf14_foot = tb14_foot.text_frame
    
    p_ty = tf14_foot.paragraphs[0]
    p_ty.text = "Thank You! Questions & Discussion"
    p_ty.font.name = FONT_HEADING
    p_ty.font.size = Pt(13.5)
    p_ty.font.bold = True
    p_ty.font.color.rgb = WHITE
    
    p_cnt = tf14_foot.add_paragraph()
    p_cnt.text = "Rishabh Dehariya (23IM10028)  ·  Dept. of Industrial & Systems Engineering, IIT Kharagpur  ·  Supervisor: Prof. Sarada Prasad Sarmah"
    p_cnt.font.name = FONT_BODY
    p_cnt.font.size = Pt(11)
    p_cnt.font.color.rgb = TEXT_LIGHT
    p_cnt.space_before = Pt(2)

    set_speaker_notes(s14,
        "In conclusion, Checkpoint 2 successfully demonstrates that PharmaSense can systematically disentangle 3-drug polypharmacy "
        "signals from spontaneous FDA reports. We have built a clean, working prototype on localhost with solved mathematical foundations, "
        "positioning PharmaSense as an affordable, accessible solution for hospital review boards and researchers. "
        "Thank you, and I look forward to your valuable questions and feedback."
    )

    prs.save(OUTPUT_FILE)
    print(f"Successfully generated presentation at {OUTPUT_FILE}")

if __name__ == "__main__":
    build_presentation()
