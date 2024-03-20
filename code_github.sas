*subset input if necessary;
data temp.diagnosis_2018;
	set sas_in.diagnosis(where=("20190101">date>="20170101"));
run; 

*get test dataset;
proc sql;
	create table temp.diagnosis_2018_ids as
    select count(distinct patient_id) as NumUniqueID
    from temp.diagnosis_2018;
quit;
proc freq data=temp.diagnosis_2018; 
tables patient_id/out=temp.diagnosis_2018_ids;
run;

data temp.diagnosis_2018_test;
    if _n_ = 1 then do;
		*only have 10000 patients;
        if 0 then set temp.diagnosis_2018_ids(obs=20000 keep=patient_id); 
        declare hash id(dataset: 'temp.diagnosis_2018_ids(obs=20000 keep=patient_id)');
        id.definekey('patient_id');
		id.definedata('patient_id');
        id.definedone();   
	end;

    set temp.diagnosis_2018 (where=(code_system="ICD-10-CM"));	

	if id.find() = 0 then output temp.diagnosis_2018_test;
run;

****************************************************************************************************************************************;
%let obs=max;
%let diagnosis_input=diagnosis_2018_test;
%let recur_time=30;
%let look_back=365;
****************************************************************************************************************************************;
*MAIN DATASET;
*get diagnosis for tia, strokes and get the main dataset by patient_id and date;
data temp.patient_strokes(rename=(date=stroke_date) drop=_date ethnicity year_of_birth year_of_dia);	
    if _n_ = 1 then do;
		*get demographics of patients;
        if 0 then set sas_in.patient(keep=patient_id sex race ethnicity marital_status year_of_birth patient_regional_location); 
        declare hash id(dataset: 'sas_in.patient(keep=patient_id sex race ethnicity marital_status year_of_birth patient_regional_location)');
        id.definekey('patient_id');
		id.definedata('sex', 'race', 'ethnicity', 'marital_status', 'year_of_birth', 'patient_regional_location');
        id.definedone();   
	end;
	call missing (sex, race, ethnicity, marital_status, year_of_birth, patient_regional_location);

	set temp.&diagnosis_input.(obs=&obs. keep=patient_id code code_system date rename=(date=_date));
		format patient_id $200. stroke_subtype $20. date date9.;

		date=input(_date,yymmdd8.);

		*tia;
		if substr(code,1,3)="G45" then stroke_subtype="TIA";
		*stroke;
		if substr(code,1,3)="I63" then stroke_subtype="IS";
		if substr(code,1,3)="I64" then stroke_subtype="Undetermined";
		if substr(code,1,3)="I61" then stroke_subtype="ICH";
		if substr(code,1,3)="I60" then stroke_subtype="SAH";	
		
		*clean for demographics;
		if id.find() = 0 then do;
			year_of_dia=substr(_date,1,4);
			age=year_of_dia-year_of_birth;
			if ethnicity="Hispanic or Latino" then race="Hispanic or Latino";
		end;
		
		if stroke_subtype^=" " then output temp.patient_strokes;
run;

*dedup for main dataset by patient_id, date, and stroke subtype;
proc sort data=temp.patient_strokes out=temp.patient_strokes_dedup nodupkey;
    by patient_id stroke_date stroke_subtype;
run;

*get recurrence;
proc sql;
    create table temp.recur_&recur_time.(where=(0<time_to_recur<=&recur_time.)) as
    select a.patient_id, 
           a.stroke_date, 
           a.stroke_subtype, 
           b.stroke_subtype as recur_subtype,
           (case when a.stroke_date < b.stroke_date then b.stroke_date - a.stroke_date else . end) as time_to_recur/*,
           (case when a.stroke_date >= b.stroke_date then a.stroke_date - b.stroke_date else . end) as time_to_dia*/
    from temp.patient_strokes_dedup as a, temp.patient_strokes_dedup as b
    where a.patient_id = b.patient_id /*and (a.stroke_date ne b.stroke_date or a.stroke_subtype ne b.stroke_subtype)*/;
quit;

*merge to get stroke event#;
data temp.main_dataset_1(drop=recur_subtype time_to_recur code_system);	
    if _n_ = 1 then do;
		*merge to recur;
        if 0 then set temp.recur_&recur_time.; 
        declare hash re(dataset: "temp.recur_&recur_time.");
        re.definekey('patient_id','stroke_subtype','stroke_date');
		re.definedata('recur_subtype');
        re.definedone();   
	end;
	call missing(recur_subtype);

	set temp.patient_strokes_dedup;
		
		stroke_event_id=_n_;
		code_short=substr(code,1,3);
		*get recur var;
		length recur recur_same 8.;
		recur=0; recur_same=0;
		if re.find() = 0 then do;
			recur=1;
			if recur_subtype=stroke_subtype then recur_same=1;
		end;		
run;


*MASTER DIAGNOSIS DATA;
*ADDED stroke histroy----------NEED TO MAKE IT FLEXIBLE!!;
data temp.stork_his;
    if _n_ = 1 then do;
		*only have 10000 patients;
        if 0 then set temp.diagnosis_2018_ids(obs=20000 keep=patient_id); 
        declare hash id(dataset: 'temp.diagnosis_2018_ids(obs=20000 keep=patient_id)');
        id.definekey('patient_id');
		id.definedata('patient_id');
        id.definedone();   
	end;
	set sas_in.diagnosis(keep=patient_id code code_system date
						where=(date<"20170101" and code_system="ICD-10-CM"));

	if id.find() = 0 then do;
		if substr(code,1,3) in ("G45","I63","I64","I61","I60") then output temp.stork_his;
	end;
run;

*dedup for diagnosis master dataset by first 3 digits of icd10, patient_id, date;
data temp.all_diagnosis(drop=_date where=(substr(code_short,1,1)^="0"));
	set temp.&diagnosis_input.(keep=patient_id code code_system date rename=(date=_date))
		temp.stork_his (rename=(date=_date));
		format patient_id $200. code_short $10. dia_date date9.;

		dia_date=input(_date,yymmdd8.);

		code_short=substr(code,1,3);
run;
proc sort data=temp.all_diagnosis out=temp.all_diagnosis_dedup (keep=patient_id dia_date code_short code) nodupkey;
    by patient_id dia_date code;
run;
data temp.all_diagnosis_dedup;
	set temp.all_diagnosis_dedup;
		dia_event_id=_n_;
run;

*merge to stroke event;
proc sql;
    create table temp.main_dataset_2 as
    select a.patient_id,
		   a.stroke_event_id, 
		   b.dia_event_id,
		   b.code_short as code_short_dia,
		   b.code as code_dia,
		   a.code_short as code_short_stroke,
           a.stroke_date,
		   b.dia_date
    from temp.main_dataset_1 as a, temp.all_diagnosis_dedup as b
    where a.patient_id = b.patient_id;
quit;

data temp.main_dataset_2(keep=stroke_event_id code_short_dia code_dia time_dur time_sort);
	set temp.main_dataset_2;
		if code_short_dia=code_short_stroke and stroke_date=dia_date then delete;
		time_dur=stroke_date-dia_date;
		*time sort indicator for dedup;
		time_sort=abs(time_dur);
		*for now, only comobidity (within 365 days);
		if &look_back.>=time_dur>=0 then output=1;
		if substr(code_dia,1,3) in ("G45","I63","I64","I61","I60") then output=1;
		if output=1 then output;
run;

*keep the closest diagnosis;
proc sort data=temp.main_dataset_2;
    by stroke_event_id code_dia time_sort;
run;
data temp.main_dataset_2_dedup (keep=stroke_event_id code_short_dia code_dia time_dur);
	set temp.main_dataset_2;
	by stroke_event_id code_dia time_sort;
	if first.code_dia then output;
run;

*long to wide;
proc transpose data=temp.main_dataset_2_dedup out=temp.main_dataset_2_wide_3(drop=_NAME_) prefix=icd10_;
    by stroke_event_id;
    id code_dia;
    var time_dur;
run;

*MERGE 2 MAIN FILES;
data temp.main_dataset_final_3_full_icd10;
merge temp.main_dataset_2_wide_3 temp.main_dataset_1;
by stroke_event_id;
run;


*DESCRIPTIVE RESULTS;
*a descriptive table: #of patients, #of stroke events, #of ICD-10s, #of recurrence� demographics ;
proc sql;
    create table temp.n_patients_events as
    select count(distinct patient_id) as n_patients,
           count(distinct stroke_event_id) as n_events,
           sum(case when recur=1 then 1 else 0 end) as n_recur
    from temp.main_dataset_final_3_full_icd10;
quit;

proc sql;
    create table temp.n_icd10 as
    select count(distinct code_dia) as n_icd10,
           count(distinct case when substr(code_dia,1,3) in ('E65','E66','E67','E68') then code_dia else " " end) as n_obesity,
           count(distinct case when substr(code_dia,1,3) in ('E78') then code_dia else " " end) as n_hyperlip,
           count(distinct case when substr(code_dia,1,3) in ('E10','E11','E12','E13','E14') then code_dia else " " end) as n_diabete,
           count(distinct case when substr(code_dia,1,3) in ('I10','I11','I12','I13','I15') then code_dia else " " end) as n_hypertn,
           count(distinct case when substr(code_dia,1,3) in ('F17','Z72') then code_dia else " " end) as n_smoke,
           count(distinct case when substr(code_dia,1,3) in ('F10') then code_dia else " " end) as n_alcohol,
           count(distinct case when substr(code_dia,1,3) in ('I48') then code_dia else " " end) as n_af,
           count(distinct case when substr(code_dia,1,3) in ('I20','I21','I22','I24','I25') then code_dia else " " end) as n_hd,
           count(distinct case when substr(code_dia,1,3) in ('I63','I64','I60','G45','I61') then code_dia else " " end) as n_stroke
    from temp.main_dataset_2_dedup;
quit;

proc sort data=temp.main_dataset_final_3_full_icd10 out=patients (keep=patient_id age sex race marital_status patient_regional_location) nodupkey;
    by patient_id;
run;

Proc freq data=patients;
tables sex race marital_status patient_regional_location; 
run;

proc means data=patients;
var age;
run;

