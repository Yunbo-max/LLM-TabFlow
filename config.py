import argparse
import os
from dotenv import load_dotenv

load_dotenv()

def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="dataset to be used", type=str, default='icustays')
    parser.add_argument("--model", help="generative model", type=str, default='deepseek')
    parser.add_argument("--temp", type=float, default=0.1)
    parser.add_argument("--max_tok", type=int, default=1000)
    return parser.parse_args()

def get_api_key(model):
    if model in ('gpt', 'gpt-4'):
        return os.getenv("OPENAI_API_KEY")
    elif model == 'deepseek':
        return os.getenv("DEEPSEEK_API_KEY")
    elif model == 'claude':
        return os.getenv("ANTHROPIC_API_KEY")
    elif model == 'llama':
        return os.getenv("TOGETHER_API_KEY")
    elif model in ('qwen', 'minimax'):
        return os.getenv("SILICONFLOW_API_KEY")
    else:
        return None

def get_column_descriptions(data_name):
    descriptions = {
        'dataco': """
        payment_type,:  Type of transaction made
        profit_per_order,:  Earnings per order placed
        sales_per_customer,:  Total sales per customer made per customer
        category_id,:  Product category code
        category_name,:  Description of the product category
        customer_city,:  City where the customer made the purchase
        customer_country,:  Country where the customer made the purchase
        customer_id,:  Customer ID
        customer_segment,:  Types of Customers: Consumer , Corporate , Home Office
        customer_state,:  State to which the store where the purchase is registered belongs
        customer_zipcode,:  Customer Zipcode
        department_id,:  Department code of store
        department_name,:  Department name of store
        latitude,:  Latitude corresponding to location of store
        longitude,:  Longitude corresponding to location of store
        market,:  Market to where the order is delivered : Africa , Europe , LATAM , Pacific Asia , USCA
        order_city,:  Destination city of the order
        order_country,:  Destination country of the order
        order_customer_id,:  Customer order code
        order_date,:  Date on which the order is made
        order_id,:  Order code
        order_item_cardprod_id,:  Product code generated through the RFID reader
        order_item_discount,:  Order item discount value
        order_item_discount_rate,:  Order item discount percentage
        order_item_id,:  Order item code
        order_item_product_price,:  Price of products without discount
        order_item_profit_ratio,:  Order Item Profit Ratio
        order_item_quantity,:  Number of products per order
        sales,:  Value in sales
        order_item_total_amount,:  Total amount per order
        order_profit_per_order,:  Order Profit Per Order
        order_region,:  Region of the world where the order is delivered :  Southeast Asia ,South Asia ,Oceania ,Eastern Asia, West Asia , West of USA , US Center , West Africa, Central Africa ,North Africa ,Western Europe ,Northern , Caribbean , South America ,East Africa ,Southern Europe , East of USA ,Canada ,Southern Africa , Central Asia ,  Europe , Central America, Eastern Europe , South of  USA
        order_state,:  State of the region where the order is delivered
        order_status,:  Order Status : COMPLETE , PENDING , CLOSED , PENDING_PAYMENT ,CANCELED , PROCESSING ,SUSPECTED_FRAUD ,ON_HOLD ,PAYMENT_REVIEW
        product_card_id,:  Product code
        product_category_id,:  Product category code
        product_name,:  Product Name
        product_price,:  Product Price
        shipping_date,:  Exact date and time of shipment
        shipping_mode,:  The following shipping modes are presented : Standard Class , First Class , Second Class , Same Day
        label,:  Target label indicating outcome of the order (e.g., success or failure)
        """,

        'adult': """
        age: Continuous. Age of the individual.
        workclass: Categorical. Type of employment (Private, Self-emp, Government, etc.).
        fnlwgt: Continuous. Final weight (sampling weight representing population count).
        education: Categorical. Highest education level (Bachelors, HS-grad, etc.).
        educational-num: Continuous. Numerical representation of education level.
        marital-status: Categorical. Marital status (Married, Divorced, Never-married, etc.).
        occupation: Categorical. Type of occupation (Tech-support, Prof-specialty, etc.).
        relationship: Categorical. Family relationship (Wife, Husband, Not-in-family, etc.).
        race: Categorical. Race (White, Black, Asian-Pac-Islander, etc.).
        sex: Categorical. Biological sex (Male, Female).
        capital-gain: Continuous. Capital gains income.
        capital-loss: Continuous. Capital losses.
        hours-per-week: Continuous. Hours worked per week.
        native-country: Categorical. Country of origin (United-States, Mexico, India, etc.).
        income: Binary. Income class (<=50K or >50K).
        """,

        'icustays': """
        SUBJECT_ID: Integer. Unique identifier for a patient. Consistent across all hospitalizations.
        HADM_ID: Integer. Unique identifier for a patient's hospital admission.
        ICUSTAY_ID: Integer. Unique identifier for a patient's ICU stay (multiple stays possible per HADM_ID).
        DBSOURCE: Categorical. Source database for the record: 'carevue' (2001-2008) or 'metavision' (2008-2012).
        FIRST_CAREUNIT: Categorical. First ICU type the patient was admitted to (e.g., 'MICU', 'SICU', 'CCU').
        LAST_CAREUNIT: Categorical. Last ICU type before transfer/discharge (matches FIRST_CAREUNIT if no transfer).
        FIRST_WARDID: Integer. Physical location ID of the first ICU unit (technical 'ward' identifier).
        LAST_WARDID: Integer. Physical location ID of the last ICU unit.
        INTIME: Datetime. When the patient entered the ICU (timestamp with timezone).
        OUTTIME: Datetime. When the patient left the ICU (NULL if still in ICU).
        LOS: Float. Length of ICU stay in fractional days (calculated from INTIME and OUTTIME).
        """,


    
     'transfers':
     """
        SUBJECT_ID: Integer. Unique identifier for a patient. Remains consistent across multiple hospitalizations.
        HADM_ID: Integer. Unique identifier for a single hospital admission. A patient (SUBJECT_ID) can have multiple HADM_IDs.
        ICUSTAY_ID: Integer. Unique identifier for a patient's ICU stay. Multiple ICU stays can exist within one HADM_ID.
        DBSOURCE: Categorical. Indicates the source database: 'carevue' (2001–2008) or 'metavision' (2008–2012). Data formats and availability may differ between sources.
        EVENTTYPE: Categorical. Type of event in patient trajectory: 'admit', 'transfer', or 'discharge'.
        PREV_CAREUNIT: Categorical. ICU care unit the patient was in before the current transfer event.
        CURR_CAREUNIT: Categorical. ICU care unit the patient is transferred into. Defined by ward ICU cost center.
        PREV_WARDID: Integer. Physical ward ID from which the patient was transferred. Includes ICUs and non-ICUs.
        CURR_WARDID: Integer. Physical ward ID to which the patient was transferred. Corresponds to CURR_CAREUNIT.
        INTIME: Datetime. Timestamp when the patient was transferred into CURR_CAREUNIT.
        OUTTIME: Datetime. Timestamp when the patient left CURR_CAREUNIT. May be null if the patient is still admitted.
        LOS: Float. Length of stay (in fractional days) in CURR_CAREUNIT, computed from INTIME and OUTTIME.
        """,
    
    'admissions': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique identifier for a patient. Remains consistent across multiple hospitalizations.
        HADM_ID: Integer. Unique identifier for a single hospital admission. A patient (SUBJECT_ID) can have multiple HADM_IDs.
        ADMITTIME: Datetime. Date and time when the patient was admitted to the hospital.
        DISCHTIME: Datetime. Date and time when the patient was discharged from the hospital.
        DEATHTIME: Datetime. Time of in-hospital death (null if patient survived). Usually matches DISCHTIME if present.
        ADMISSION_TYPE: Categorical. Type of admission: 'ELECTIVE', 'URGENT', 'NEWBORN', or 'EMERGENCY'.
        ADMISSION_LOCATION: Categorical. Previous location before hospital arrival (e.g., 'EMERGENCY ROOM ADMIT', 'TRANSFER FROM HOSP/EXTRAM').
        DISCHARGE_LOCATION: Categorical. Location after discharge (e.g., 'HOME', 'REHAB', 'DIED').
        INSURANCE: Categorical. Patient's insurance type (e.g., 'Medicare', 'Private').
        LANGUAGE: Categorical. Patient's primary language (e.g., 'ENGLISH', 'SPANISH').
        RELIGION: Categorical. Patient's religious affiliation (e.g., 'CATHOLIC', 'PROTESTANT').
        MARITAL_STATUS: Categorical. Patient's marital status (e.g., 'MARRIED', 'SINGLE').
        ETHNICITY: Categorical. Patient's ethnic background (e.g., 'WHITE', 'BLACK', 'HISPANIC').
        EDREGTIME: Datetime. Time registered in emergency department (null if not via ED).
        EDOUTTIME: Datetime. Time discharged from emergency department (null if not via ED).
        DIAGNOSIS: Text. Preliminary free-text diagnosis on admission (often vague, not standardized).
        HOSPITAL_EXPIRE_FLAG: Boolean. 1 if patient died in-hospital, 0 if survived to discharge.
        HAS_CHARTEVENTS_DATA: Boolean. 1 if chart events data exists for this admission.
        """,

    
    'callout':"""
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique identifier for a patient. Links to PATIENTS table.
        HADM_ID: Integer. Unique identifier for a hospital admission. Links to ADMISSIONS table.
        SUBMIT_WARDID: Integer. Ward ID where the discharge request was submitted from.
        SUBMIT_CAREUNIT: Categorical. ICU type of the submitting ward (if applicable).
        CURR_WARDID: Integer. Current ward ID where patient resides before transfer.
        CURR_CAREUNIT: Categorical. Current ICU type (all patients are in ICU during callout).
        CALLOUT_WARDID: Integer. Target ward ID for discharge (0=Home, 1=First available ward).
        CALLOUT_SERVICE: Categorical. Service under which patient should be discharged.
        REQUEST_TELE: Boolean. If telemetry monitoring was requested for next ward.
        REQUEST_RESP: Boolean. If respiratory isolation was requested.
        REQUEST_CDIFF: Boolean. If C.difficile precautions were requested.
        REQUEST_MRSA: Boolean. If MRSA precautions were requested.
        REQUEST_VRE: Boolean. If VRE precautions were requested.
        CALLOUT_STATUS: Categorical. Status of callout ('Active' or 'Inactive').
        CALLOUT_OUTCOME: Categorical. Final outcome ('Discharged' or 'Cancelled').
        DISCHARGE_WARDID: Integer. Actual ward ID discharged to (0=Home).
        ACKNOWLEDGE_STATUS: Categorical. Response to callout ('Acknowledged', 'Revised', etc.).
        CREATETIME: Datetime. When the callout was initiated.
        UPDATETIME: Datetime. Last update time for the callout event.
        ACKNOWLEDGETIME: Datetime. When the callout was first acknowledged.
        OUTCOMETIME: Datetime. When the final outcome occurred.
        FIRSTRESERVATIONTIME: Datetime. First ward reservation time.
        CURRENTRESERVATIONTIME: Datetime. Current ward reservation time.
        """,
    

    'caregivers': """
        ROW_ID: Integer. Unique identifier for the row.
        CGID: Integer. Unique identifier for a caregiver. Links to CHARTEVENTS table.
        LABEL: Categorical. Type of caregiver (e.g., 'RN', 'MD', 'PharmD'). Free-text field with potential variants/typos.
        DESCRIPTION: Categorical. Additional structured information about the caregiver (17 unique values in MIMIC-III v1.0).
        """,

    
    'chartevents':"""
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Unique hospital admission identifier. Links to ADMISSIONS table.
        ICUSTAY_ID: Integer. Unique ICU stay identifier. Links to ICUSTAYS table.
        ITEMID: Integer. Measurement type identifier. Links to D_ITEMS table.
        CHARTTIME: Datetime. Time when observation was made (closest to actual measurement time).
        STORETIME: Datetime. Time when observation was validated/input by clinical staff.
        CGID: Integer. Caregiver identifier who validated measurement. Links to CAREGIVERS table.
        VALUE: Text. Measured value in string format (may contain numeric or text data).
        VALUENUM: Numeric. Numeric representation of VALUE when applicable (null for non-numeric data).
        VALUEUOM: Text. Unit of measurement for the value.
        WARNING: Boolean. (Metavision only) Indicates if a warning was raised for the value.
        ERROR: Boolean. (Metavision only) Indicates if an error occurred during measurement.
        RESULTSTATUS: Text. (CareVue only) Measurement type: 'Manual' or 'Automatic'.
        STOPPED: Text. (CareVue only) Indicates if measurement was stopped.
        """,
    

    'cptevents': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Unique hospital admission identifier. Links to ADMISSIONS table.
        COSTCENTER: Categorical. Billing cost center ('ICU' or 'Resp' for respiratory).
        CHARTDATE: Datetime. Date when procedure was performed.
        CPT_CD: Text. Original CPT procedure code (alphanumeric).
        CPT_NUMBER: Integer. Numeric portion of CPT code for range comparisons.
        CPT_SUFFIX: Text. Non-numeric suffix portion of CPT code.
        TICKET_ID_SEQ: Integer. Sequence order of the CPT code.
        SECTIONHEADER: Text. Broad category header for the CPT code.
        SUBSECTIONHEADER: Text. More specific subcategory header.
        DESCRIPTION: Text. Detailed description of procedure (especially for respiratory codes).
        """,
    
    'd_cpt': """
        ROW_ID: Integer. Unique identifier for the row.
        CATEGORY: Integer. Numeric category identifier for the CPT code group.
        SECTIONRANGE: Text. Range of CPT codes included in this section (text format).
        SECTIONHEADER: Text. Description of the section (e.g., 'Surgery', 'Anesthesia').
        SUBSECTIONRANGE: Text. Range of CPT codes included in this subsection (text format).
        SUBSECTIONHEADER: Text. Detailed description of the subsection.
        CODESUFFIX: Text. Suffix used for codes in this subsection.
        MINCODEINSUBSECTION: Integer. Minimum numeric value in the subsection range (for joining).
        MAXCODEINSUBSECTION: Integer. Maximum numeric value in the subsection range (for joining).


        """,
    
    'd_icd_diagnoses': """
        ROW_ID: Integer. Unique identifier for the row.
        ICD9_CODE: Text. International Classification of Diseases Version 9 (ICD-9) code. Links to DIAGNOSES_ICD table.
        SHORT_TITLE: Text. Brief description of the diagnosis (50 characters max).
        LONG_TITLE: Text. Detailed description of the diagnosis (300 characters max).
        """,
    
     'd_icd_procedures': """
        ROW_ID: Integer. Unique identifier for the row.
        ICD9_CODE: Text. International Classification of Diseases Version 9 (ICD-9) procedure code. Links to PROCEDURES_ICD table.
        SHORT_TITLE: Text. Concise procedure description (50 characters max).
        LONG_TITLE: Text. Detailed procedure description (300 characters max).
        """,
                        
    'd_items':"""
        ROW_ID: Integer. Unique identifier for the row.
        ITEMID: Integer. Unique measurement identifier. Links to multiple event tables.
        LABEL: Text. Description of the measurement concept (200 characters max).
        ABBREVIATION: Text. Common abbreviation for the label (Metavision only, 100 chars max).
        DBSOURCE: Categorical. Source database: 'carevue' or 'metavision'.
        LINKSTO: Text. Target event table containing this ITEMID (e.g., 'chartevents').
        CATEGORY: Text. Type of measurement (e.g., 'ABG', 'IV Medication').
        UNITNAME: Text. Unit of measurement (when applicable).
        PARAM_TYPE: Categorical. Data type: 'Date', 'Numeric', or 'Text'.
        CONCEPTID: Integer. Concept identifier for grouping equivalent ITEMIDs across sources.
        """,

    'd_labitems': """
        ROW_ID: Integer. Unique identifier for the row.
        ITEMID: Integer. Unique laboratory measurement identifier. Links to LABEVENTS table.
        LABEL: Text. Description of the lab test (100 characters max).
        FLUID: Categorical. Substance measured (e.g., 'BLOOD', 'URINE', 'CSF').
        CATEGORY: Categorical. Test category (e.g., 'HEMATOLOGY', 'CHEMISTRY', 'MICROBIOLOGY').
        LOINC_CODE: Text. Standardized LOINC code for the test (when available).
        """,
    
    'datetimeevents': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Unique hospital admission identifier. Links to ADMISSIONS table.
        ICUSTAY_ID: Integer. Unique ICU stay identifier. Links to ICUSTAYS table.
        ITEMID: Integer. Measurement type identifier. Links to D_ITEMS table.
        CHARTTIME: Datetime. When observation was charted (closest to measurement time).
        STORETIME: Datetime. When observation was validated/input by clinical staff.
        CGID: Integer. Caregiver who validated measurement. Links to CAREGIVERS table.
        VALUE: Datetime. The actual date/time value being recorded.
        VALUEUOM: Text. Unit of measurement for the date value (usually 'date').
        WARNING: Boolean. (Metavision only) Indicates if a warning was raised.
        ERROR: Boolean. (Metavision only) Indicates if an error occurred.
        RESULTSTATUS: Text. (CareVue only) Measurement type: 'Manual' or 'Automatic'.
        STOPPED: Text. (CareVue only) Indicates if measurement was stopped.
        """,
    
    'diagnoses_icd': """
        ROW_ID: Integer. Unique identifier for the row (not null).
        SUBJECT_ID: Integer. Unique patient identifier (not null). Links to PATIENTS table.
        HADM_ID: Integer. Unique hospital admission identifier (not null). Links to ADMISSIONS table.
        SEQ_NUM: Integer. Diagnosis priority sequence (1=primary, higher numbers=secondary).
        ICD9_CODE: Text. ICD-9 diagnosis code (10 characters max). Links to D_ICD_DIAGNOSES table.
        """,

    'drgcodes': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier (not null). Links to PATIENTS table.
        HADM_ID: Integer. Unique hospital admission identifier (not null). Links to ADMISSIONS table.
        DRG_TYPE: Categorical. Type of DRG code: 'HCFA', 'MS', or 'APR'.
        DRG_CODE: Text. Diagnosis-Related Group code (20 characters max).
        DESCRIPTION: Text. Detailed description of DRG code (300 characters max).
        DRG_SEVERITY: Integer. (APR-DRG only) Severity level (1-4).
        DRG_MORTALITY: Integer. (APR-DRG only) Mortality risk level (1-4).
        """,
    
    'inputevents_cv': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Unique hospital admission identifier. Links to ADMISSIONS table.
        ICUSTAY_ID: Integer. Unique ICU stay identifier. Links to ICUSTAYS table.
        CHARTTIME: Datetime. Time when input was charted (end time for amounts, start time for rates).
        ITEMID: Integer. Medication/fluid identifier. Links to D_ITEMS table (CareVue items <220000).
        AMOUNT: Numeric. Quantity of substance administered.
        AMOUNTUOM: Text. Unit of measurement for amount.
        RATE: Numeric. Administration rate of substance.
        RATEUOM: Text. Unit of measurement for rate.
        STORETIME: Datetime. When observation was validated/input by staff.
        CGID: Integer. Caregiver who validated input. Links to CAREGIVERS table.
        ORDERID: Integer. Groups items in same solution (e.g., drug + carrier fluid).
        LINKORDERID: Integer. Links order changes across time.
        STOPPED: Text. Whether infusion was stopped ('Stopped'/'Running').
        NEWBOTTLE: Boolean. If a new solution preparation was hung.
        ORIGINALAMOUNT: Numeric. Original amount in parent solution.
        ORIGINALAMOUNTUOM: Text. Original amount unit.
        ORIGINALROUTE: Text. Original administration route.
        ORIGINALRATE: Numeric. Original administration rate.
        ORIGINALRATEUOM: Text. Original rate unit.
        ORIGINALSITE: Text. Original administration site.
        """,
    
    'inputevents_mv': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Unique hospital admission identifier. Links to ADMISSIONS table.
        ICUSTAY_ID: Integer. Unique ICU stay identifier. Links to ICUSTAYS table.
        STARTTIME: Datetime. When administration began.
        ENDTIME: Datetime. When administration ended (boluses show as 1 minute duration).
        ITEMID: Integer. Medication/fluid identifier. Links to D_ITEMS table (Metavision items >220000).
        AMOUNT: Numeric. Quantity of substance administered between STARTTIME-ENDTIME.
        AMOUNTUOM: Text. Unit of measurement for amount.
        RATE: Numeric. Administration rate between STARTTIME-ENDTIME.
        RATEUOM: Text. Unit of measurement for rate.
        STORETIME: Datetime. When observation was validated/input by staff.
        CGID: Integer. Caregiver who validated input. Links to CAREGIVERS table.
        ORDERID: Integer. Groups items in same solution (e.g., drug + carrier fluid).
        LINKORDERID: Integer. Links order changes across time.
        ORDERCATEGORYNAME: Text. Primary category of order (e.g., 'Continuous IV').
        SECONDARYORDERCATEGORYNAME: Text. Secondary category of order.
        ORDERCOMPONENTTYPEDESCRIPTION: Text. Component role ('Base', 'Additive', etc).
        ORDERCATEGORYDESCRIPTION: Text. Description of order category.
        PATIENTWEIGHT: Numeric. Patient weight in kilograms at time of administration.
        TOTALAMOUNT: Numeric. Total amount in the IV bag/solution.
        TOTALAMOUNTUOM: Text. Unit for total amount.
        ISOPENBAG: Boolean. If solution came from an open bag.
        CONTINUEINNEXTDEPT: Boolean. If administration continued after transfer.
        CANCELREASON: Integer. Reason code if order was canceled.
        STATUSDESCRIPTION: Text. Final status ('Changed', 'FinishedRunning', etc).
        COMMENTS_STATUS: Text. Edit/cancel status if order was modified.
        COMMENTS_TITLE: Text. Job title of caregiver who modified order.
        COMMENTS_DATE: Datetime. When order was modified.
        ORIGINALAMOUNT: Numeric. Original drug amount in solution at STARTTIME.
        ORIGINALRATE: Numeric. Originally prescribed administration rate.
        """,
    
    'labevents': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Hospital admission identifier (null for outpatient data). Links to ADMISSIONS table.
        ITEMID: Integer. Laboratory test identifier. Links to D_LABITEMS table.
        CHARTTIME: Datetime. Time when specimen was collected (not result time).
        VALUE: Text. Laboratory result in string format.
        VALUENUM: Numeric. Numeric representation of result when applicable.
        VALUEUOM: Text. Unit of measurement for the result.
        FLAG: Text. Abnormal result indicator (e.g., 'abnormal', 'high', 'low').

        """,
    

    'microbiologyevents': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Hospital admission identifier. Links to ADMISSIONS table.
        CHARTDATE: Date. Date when culture was collected (always available).
        CHARTTIME: Datetime. Time when culture was collected (null if time unknown).
        SPEC_ITEMID: Integer. Specimen type identifier. Links to D_ITEMS table.
        SPEC_TYPE_DESC: Text. Description of specimen type (e.g., 'BLOOD', 'URINE').
        ORG_ITEMID: Integer. Organism identifier (null if no growth). Links to D_ITEMS table.
        ORG_NAME: Text. Name of organism cultured (null if no growth).
        ISOLATE_NUM: Integer. Isolate number for antibiotic testing (starts at 1).
        AB_ITEMID: Integer. Antibiotic tested identifier. Links to D_ITEMS table.
        AB_NAME: Text. Name of antibiotic tested.
        DILUTION_TEXT: Text. Text description of dilution concentration.
        DILUTION_COMPARISON: Text. Comparison operator for dilution (e.g., '>', '<=').
        DILUTION_VALUE: Numeric. Numeric value of dilution concentration.
        INTERPRETATION: Text. Sensitivity result ('S', 'R', 'I', or 'P').
        """,

    'noteevents':"""
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Hospital admission identifier (null for outpatient notes). Links to ADMISSIONS table.
        CHARTDATE: Date. Date when note was created (always available).
        CHARTTIME: Datetime. Time when note was created (null for some categories).
        STORETIME: Datetime. When note was saved to system (null for some categories).
        CATEGORY: Text. Note type (e.g., 'Discharge summary', 'Radiology').
        DESCRIPTION: Text. Note subtype (e.g., 'Report', 'Addendum').
        CGID: Integer. Caregiver who created note. Links to CAREGIVERS table.
        ISERROR: Boolean. If note was flagged as containing errors.
        TEXT: Text. Full content of the clinical note (may contain newlines).
        """,
    
    'outputevents': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Unique hospital admission identifier. Links to ADMISSIONS table.
        ICUSTAY_ID: Integer. Unique ICU stay identifier. Links to ICUSTAYS table.
        CHARTTIME: Datetime. Time when output was measured/recorded.
        ITEMID: Integer. Output type identifier. Links to D_ITEMS table (CareVue: 40000-49999, Metavision: >220000).
        VALUE: Numeric. Quantity of output measured.
        VALUEUOM: Text. Unit of measurement for output.
        STORETIME: Datetime. When observation was validated/input by staff.
        CGID: Integer. Caregiver who validated output. Links to CAREGIVERS table.
        STOPPED: Text. Whether output measurement was stopped.
        NEWBOTTLE: Boolean. If a new collection container/bag was used.
        ISERROR: Boolean. (Metavision only) If output was flagged as erroneous.
        """,
    

    'patients': """
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier (primary key). Links to all patient-related tables.
        GENDER: Categorical. Patient's biological sex ('M' or 'F').
        DOB: Datetime. Date of birth (shifted 300 years back for patients >89 years old).
        DOD: Datetime. Date of death (combines hospital and SSN records, hospital takes precedence).
        DOD_HOSP: Datetime. Date of death recorded in hospital database.
        DOD_SSN: Datetime. Date of death from Social Security Death Index.
        EXPIRE_FLAG: Boolean. 1 if patient died (DOD not null), 0 otherwise.
    
        """,

    'prescriptions':"""
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Hospital admission identifier. Links to ADMISSIONS table.
        ICUSTAY_ID: Integer. ICU stay identifier. Links to ICUSTAYS table.
        STARTDATE: Datetime. When prescription became active.
        ENDDATE: Datetime. When prescription ended (may be null).
        DRUG_TYPE: Text. Type of drug (e.g., 'MAIN', 'STANDARD').
        DRUG: Text. Name of drug as entered in the system.
        DRUG_NAME_POE: Text. Name of drug from provider order entry.
        DRUG_NAME_GENERIC: Text. Generic name of drug.
        FORMULARY_DRUG_CD: Text. Hospital formulary code.
        GSN: Text. Generic Sequence Number code.
        NDC: Text. National Drug Code.
        PROD_STRENGTH: Text. Drug strength and form (e.g., '650mg TAB').
        DOSE_VAL_RX: Text. Prescribed dose value.
        DOSE_UNIT_RX: Text. Prescribed dose unit.
        FORM_VAL_DISP: Text. Amount of drug form dispensed.
        FORM_UNIT_DISP: Text. Unit of drug form dispensed.
        ROUTE: Text. Administration route (e.g., 'PO', 'IV').
        """,
    

    'procedureevents_mv': """
        ROW_ID: Integer. Unique identifier for the row (not null).
        SUBJECT_ID: Integer. Unique patient identifier (not null). Links to PATIENTS table.
        HADM_ID: Integer. Hospital admission identifier (not null). Links to ADMISSIONS table.
        ICUSTAY_ID: Integer. ICU stay identifier. Links to ICUSTAYS table.
        STARTTIME: Datetime. When procedure began.
        ENDTIME: Datetime. When procedure ended.
        ITEMID: Integer. Procedure identifier (Metavision ITEMID >220000). Links to D_ITEMS table.
        VALUE: Numeric. Quantitative value associated with procedure.
        VALUEUOM: Text. Unit of measurement for procedure value.
        LOCATION: Text. Where procedure was performed.
        LOCATIONCATEGORY: Text. Category of procedure location.
        STORETIME: Datetime. When procedure was documented in system.
        CGID: Integer. Caregiver who performed procedure. Links to CAREGIVERS table.
        ORDERID: Integer. Groups related procedure orders.
        LINKORDERID: Integer. Links procedure order changes over time.
        ORDERCATEGORYNAME: Text. Primary category of procedure order.
        SECONDARYORDERCATEGORYNAME: Text. Secondary category of procedure order.
        ORDERCATEGORYDESCRIPTION: Text. Description of order category.
        ISOPENBAG: Boolean. If procedure used open supplies/kit.
        CONTINUEINNEXTDEPT: Boolean. If procedure continued after transfer.
        CANCELREASON: Integer. Reason code if procedure was canceled.
        STATUSDESCRIPTION: Text. Final status of procedure.
        COMMENTS_EDITEDBY: Text. Staff who edited procedure comments.
        COMMENTS_CANCELEDBY: Text. Staff who canceled procedure.
        COMMENTS_DATE: Datetime. When comments were added/modified.
        """,
    

     'procedures_icd': """
        ROW_ID: Integer. Unique identifier for the row (not null).
        SUBJECT_ID: Integer. Unique patient identifier (not null). Links to PATIENTS table.
        HADM_ID: Integer. Hospital admission identifier (not null). Links to ADMISSIONS table.
        SEQ_NUM: Integer. Procedure sequence number (indicates order performed).
        ICD9_CODE: Text. ICD-9 procedure code (10 chars max). Links to D_ICD_PROCEDURES table.
        """,
    

    'services':"""
        ROW_ID: Integer. Unique identifier for the row.
        SUBJECT_ID: Integer. Unique patient identifier. Links to PATIENTS table.
        HADM_ID: Integer. Hospital admission identifier. Links to ADMISSIONS table.
        TRANSFERTIME: Datetime. When patient was transferred between services.
        PREV_SERVICE: Categorical. Previous service type (e.g., 'MED', 'SURG').
        CURR_SERVICE: Categorical. Current service type (e.g., 'CMED', 'NSURG').
        
        # Service Type Abbreviations and Descriptions:
        # CMED   - Cardiac Medical
        # CSURG  - Cardiac Surgery  
        # DENT   - Dental
        # ENT    - Ear/Nose/Throat
        # GU     - Genitourinary
        # GYN    - Gynecological
        # MED    - Medical (General)
        # NB/NBB - Newborn
        # NMED   - Neurologic Medical  
        # NSURG  - Neurologic Surgical
        # OBS    - Obstetrics
        # ORTHO  - Orthopaedic
        # OMED   - Oncologic Medical
        # PSURG  - Plastic Surgery
        # PSYCH  - Psychiatric
        # SURG   - Surgical (General)
        # TRAUM  - Trauma
        # TSURG  - Thoracic Surgical
        # VSURG  - Vascular Surgical
        """,

    'purchasing': """
        supplier_number: Integer. Unique identifier for the supplier.
        supplier_name: Categorical. Name of the supplier company.
        order_created_date: Datetime. Date when the purchase order was created.
        planned_delivery_date: Datetime. Planned delivery date agreed upon at order creation.
        actual_delivery_date: Datetime. Actual date the goods were delivered.
        material_number: Integer. Unique identifier for the material/product ordered.
        material_description: Categorical. Description of the material/product.
        quantity: Numeric. Number of units ordered.
        unit_price: Numeric. Price per unit of the material.
        discount_rate: Numeric. Discount percentage applied to the order (0 to 1).
        net_amount: Numeric. Net purchase amount before tax (quantity × unit_price).
        gross_amount: Numeric. Gross purchase amount after discount (net_amount × (1 - discount_rate)).
        currency: Categorical. Currency of the transaction (e.g., EUR, USD).
        purchasing_group: Categorical. Internal purchasing group responsible for the order.
        plant: Categorical. Manufacturing plant or warehouse receiving the goods.
        order_status: Categorical. Status of the purchase order (e.g., Completed, Pending, Cancelled).
        payment_terms: Categorical. Payment terms agreed with supplier (e.g., Net 30, Net 60).
        delivery_indicator: Categorical. Indicates whether delivery was on time, early, or late.
        goods_receipt_date: Datetime. Date when goods were officially received and recorded.
        invoice_date: Datetime. Date of the supplier invoice.
        tax_amount: Numeric. Tax amount applied to the order.
        """
    }

    return descriptions.get(data_name, "")