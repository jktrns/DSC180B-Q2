

select
    countryname_normalized as country,
    count(distinct a.guid) as number_of_systems,
    avg(num_power_ons) as avg_number_of_dc_powerons,
    avg(duration_mins) as avg_duration
from
    reporting.system_batt_dc_events a
inner join
    reporting.system_sysinfo_unique_normalized b
on
    a.guid = b.guid
group by
    countryname_normalized
having
    count(distinct a.guid) > 100
order by
    avg_number_of_dc_powerons desc;

select * from reporting.system_sysinfo_unique_normalized;

select * from reporting.system_cpu_metadata;

select
    marketcodename,
    cpugen,
    count(distinct b.guid) as number_of_systems,
    avg(duration_mins) as avg_duration_mins_on_battery
from
    reporting.system_cpu_metadata a
inner join
    reporting.system_batt_dc_events b
on
    a.guid = b.guid
where
    cpugen != 'Unknown'
group by
    marketcodename, cpugen
having
    count(distinct b.guid) > 100;

select * from reporting.system_mods_top_blocker_hist;


select os_name, os_codename, count(*) as num_entries, count(distinct guid) as number_of_systems, count(*) *1.0/count(distinct guid) as entries_per_system
from (select a.guid, min_ts, max_ts, os_name, os_codename, dt
      from reporting.system_mods_top_blocker_hist a
               inner join
           reporting.system_os_codename_history b
           on
               a.guid = b.guid
      where a.dt between b.min_ts and b.max_ts) inn
group by os_name, os_codename
having count(distinct guid) > 10;



select
    os_name,
    os_codename,
    blocker_name,
    blocker_type,
    activity_level,
    count(distinct guid) as number_of_clients,
    avg(active_time_sec) as average_active_time_in_seconds,
    count(*) as number_of_occurences
from
(
    select
        a.guid,
        blocker_name,
        active_time_ms / 1000.0 as active_time_sec,
        activity_level,
        blocker_type,
        os_name,
        os_codename
    from
        reporting.system_mods_top_blocker_hist a
    inner join
        reporting.system_os_codename_history b
    on
        a.guid = b.guid
    and
        a.dt between b.min_ts and b.max_ts
    where
        active_time_ms > 0
) as inn
group by
    os_name, os_codename, blocker_name, blocker_type, activity_level;


select
    b.marketcodename,
    b.cpugen,
    count(distinct a.guid) as number_of_systems,
    round(avg(on_time),2) as avg_on_time,
    round(avg(off_time),2) as avg_off_time,
    round(avg(mods_time),2) as avg_modern_sleep_time,
    round(avg(sleep_time),2) as avg_sleep_time,
    round(avg(on_time+ off_time+ mods_time+ sleep_time),2) as avg_total_time,
    round(sum(on_time) *100.0 / sum(on_time+off_time+mods_time+sleep_time),2) as avg_pcnt_on_time,
    round(sum(off_time) *100.0 / sum(on_time+off_time+mods_time+sleep_time),2) as avg_pcnt_off_time,
    round(sum(mods_time) *100.0 / sum(on_time+off_time+mods_time+sleep_time),2) as avg_pcnt_mods_time,
    round(sum(sleep_time) *100.0 / sum(on_time+off_time+mods_time+sleep_time),2) as avg_pcnt_sleep_time
from
    reporting.system_on_off_suspend_time_day a
inner join
    reporting.system_cpu_metadata bselect *
from (select app_type,
             exe_name,
             count(distinct guid)                                                   as number_of_systems,
             rank() over (partition by app_type order by count(distinct guid) desc) as rank
      from reporting.system_frgnd_apps_types
      where
          exe_name not in ('restricted process', 'desktop')
      and
          app_type is not null
      group by
          app_type, exe_name
      order by number_of_systems desc) inn
where rank <= 10
order by app_type, rank asc;
on
    a.guid = b.guid
group by
    b.marketcodename, b.cpugen
having
    count(distinct a.guid) > 100;


select *
from (select app_type,
             exe_name,
             count(distinct guid)                                                   as number_of_systems,
             rank() over (partition by app_type order by count(distinct guid) desc) as rank
      from reporting.system_frgnd_apps_types
      where
          exe_name not in ('restricted process', 'desktop')
      and
          app_type is not null
      group by
          app_type, exe_name
      order by number_of_systems desc) inn
where rank <= 10
order by app_type, rank asc;



select
    app_type,
    exe_name,
    round(average_focal_sec_per_day) as average_focal_sec_per_day,
    rank
from (select app_type,
             exe_name,
             avg(totalsecfocal_day)as average_focal_sec_per_day,
             rank() over (partition by app_type order by avg(totalsecfocal_day) desc) as rank
      from reporting.system_frgnd_apps_types
      where
          exe_name not in ('restricted process', 'desktop')
      and
          app_type is not null
      group by
          app_type, exe_name
      order by average_focal_sec_per_day desc) inn
where rank <= 10
order by app_type, rank asc;

select
    app_type,
    exe_name,
    total_num_detections as total_number_of_detections,
    rank
from (select app_type,
             exe_name,
             sum(lines_per_day) as total_num_detections,
             rank() over (partition by app_type order by sum(lines_per_day) desc) as rank
      from reporting.system_frgnd_apps_types
      where
          exe_name not in ('restricted process', 'desktop')
      and
          app_type is not null
      group by
          app_type, exe_name
      order by total_num_detections desc) inn
where rank <= 10
order by app_type, rank asc;



select count(*) from university_prod.display_devices limit 100

select * from university_prod.display_devices limit 100;


select
    --connection_type,
    vendor_name,
    count(distinct guid) from university_prod.display_devices group by 1;

select
    guid,
    count(distinct vendor_name) as num_vendors
from university_prod.display_devices group by 1
order by num_vendors desc;

select * from university_prod.display_devices where guid = '03396563d3184be6831f22e6b587d20b';

select resolution_heigth::text || 'x' || resolution_width::text as resolution,
       count(distinct guid) as number_of_systems
from university_prod.display_devices
group by 1
order by number_of_systems desc

select guid, dt, sum(duration_ac + duration_dc) as total_duration from university_prod.display_devices group by guid, dt having sum(duration_ac + duration_dc) > 86400

drop table reporting.system_display_devices;
create table reporting.system_display_devices
(
    load_ts              timestamp,
    guid                 text,
    dt                   date,
    ts                   timestamp,
    display_id           bigint,
    adapter_id           bigint,
    port                 text,
    sink_index           bigint,
    connection_type      text,
    vendor_name          text,
    status               bigint,
    resolution_width     bigint,
    resolution_heigth    bigint,
    refresh_rate         double precision,
    duration_ac          bigint,
    duration_dc          bigint
);

alter table reporting.system_display_devices
    owner to postgres;

create index idx_display_devices_guid on reporting.system_display_devices (guid);

insert into reporting.system_display_devices
select
    load_ts::timestamp,
    guid,
    dt::date,
    ts::timestamp,
    display_id,
    adapter_id,
    port,
    sink_index,
    connection_type,
    vendor_name,
    status,
    resolution_width,
    resolution_heigth,
    refresh_rate,
    duration_ac,
    duration_dc
from
    university_prod.display_devices;

select
    connection_type,
    resolution_heigth::text || 'x' || resolution_width::text as resolution,
    count(distinct guid) as number_of_systems,
    round(avg(duration_ac),2) as average_duration_on_ac_in_seconds,
    round(avg(duration_dc),2) as average_duration_on_dc_in_seconds
from
    reporting.system_display_devices
where
    connection_type is not null
and
    resolution_heigth !=0
and
    resolution_width !=0
group by 1, 2
having count(distinct guid) > 50
order by
    connection_type, number_of_systems desc;

select
    distinct status, count(*) from reporting.system_display_devices group by status;

select
    vendor_name,
    count(distinct guid) as number_of_systems,
    total_number_of_systems,
    round(count(distinct guid) *100.0/ total_number_of_systems,2 )as percentage_of_systems
from
    reporting.system_display_devices
cross join
(
    select count(distinct guid) as total_number_of_systems from reporting.system_display_devices
) inn
group by vendor_name, total_number_of_systems;

select * from university_analysis_pad.data_dictionary where table_name = 'display_devices';

select count(distinct guid), count(distinct audit_zip) from university_prod.hw_pack_run_avg_pwr;

select * from university_prod.power_acdc_usage_v4_hist limit 100;

select
    guid,
    dt,
    event_name,
    ac_dc_event_name,
    proc_name_current,
    count(*) as number_of_instances,
    sum(duration_ms) as total_duration_ms
from
    university_prod.userwait_v2
group by
    guid, dt, event_name, ac_dc_event_name, proc_name_current
limit 100;

drop table reporting.system_userwait;
create table reporting.system_userwait
(
    guid text,
    dt date,
    event_name text,
    ac_dc_event_name text,
    acdc text,
    proc_name text,
    number_of_instances int,
    total_duration_ms bigint
);

alter table reporting.system_userwait
    owner to postgres;

create index idx_userwait_guid_dt on reporting.system_display_devices (guid, dt);

insert into reporting.system_userwait
select
    guid,
    dt,
    event_name,
    ac_dc_event_name,
    UPPER(substring(ac_dc_event_name,1,2)) as acdc,
    proc_name_current as proc_name,
    count(*) as number_of_instances,
    sum(duration_ms) as total_duration_ms
from
    university_prod.userwait_v2
group by
    guid, dt, event_name, ac_dc_event_name, proc_name_current
order by guid, dt;

select count(*) from reporting.system_userwait;

select * from reporting.system_userwait limit 100;

select
    proc_name,
    total_duration_sec_per_instance,
    rank
from
(
    select
         proc_name,
         sum((total_duration_ms/1000))/sum(number_of_instances) as total_duration_sec_per_instance,
         rank() over (order by sum((total_duration_ms/1000))/sum(number_of_instances) desc) as rank
    from
        reporting.system_userwait
    where
        proc_name not in ('DUMMY_PROCESS', 'DESKTOP', 'explorer.exe', 'RESTRICTED PROCESS', 'UNKNOWN')
    and
        event_name NOT in ('TOTAL_NON_WAIT_EVENTS', 'TOTAL_DISCARDED_WAIT_EVENTS')
    group by
        proc_name
    order by
        rank desc
) inn
where rank <= 10
order by rank asc;


select
    event_name,
    acdc,
    proc_name,
    round(total_duration_sec_per_instance, 2) as total_duration_sec_per_instance,
    rank
from
(
    select
        event_name,
        acdc,
        proc_name,
        sum((total_duration_ms/1000))/sum(number_of_instances) as total_duration_sec_per_instance,
        rank() over (partition by event_name, acdc order by sum((total_duration_ms/1000))/sum(number_of_instances) desc) as rank
    from
        reporting.system_userwait
    where
        proc_name not in ('DUMMY_PROCESS', 'DESKTOP', 'explorer.exe', 'RESTRICTED PROCESS', 'UNKNOWN')
    and
        event_name NOT in ('TOTAL_NON_WAIT_EVENTS', 'TOTAL_DISCARDED_WAIT_EVENTS')
    group by
        event_name, acdc, proc_name
    order by
        rank desc
) inn
where rank <= 10
order by acdc, event_name, rank asc;

select *
from
    reporting.system_userwait limit 100

select
    proc_name,
    sum(case when acdc = 'AC' then round(aggragated_duration_in_seconds/number_of_instances, 2) else 0 end) as ac_duration,
    sum(case when acdc = 'DC' then round(aggragated_duration_in_seconds/number_of_instances, 2) else 0 end) as dc_duration,
    sum(case when acdc = 'UN' then round(aggragated_duration_in_seconds/number_of_instances, 2) else 0 end) as unknown_duration
from (select procs.proc_name,
             uw.acdc,
             sum(uw.number_of_instances)     as number_of_instances,
             sum(total_duration_ms / 1000.0) as aggragated_duration_in_seconds,
             count(distinct uw.guid)         as number_of_systems
      from (select proc_name,
                   round(total_duration_sec_per_instance, 2) as total_duration_sec_per_instance,
                   rank
            from (select proc_name,
                         sum((total_duration_ms / 1000)) / sum(number_of_instances) as total_duration_sec_per_instance,
                         rank() over (order by sum((total_duration_ms / 1000)) / sum(number_of_instances) desc) as rank
                  from reporting.system_userwait
                  where proc_name not in ('DUMMY_PROCESS', 'DESKTOP', 'explorer.exe', 'RESTRICTED PROCESS', 'UNKNOWN')
                    and event_name NOT in ('TOTAL_NON_WAIT_EVENTS', 'TOTAL_DISCARDED_WAIT_EVENTS')
                  group by proc_name
                  having sum(number_of_instances) > 50
                     and count(distinct guid) > 20
                  order by rank desc) inn
            order by rank asc) procs
               inner join
           reporting.system_userwait uw
           on
               procs.proc_name = uw.proc_name
      where procs.rank <= 20
      group by procs.proc_name, uw.acdc) a
group by proc_name;

select count(*), count(distinct guid) from university_prod.hw_pack_run_avg_pwr
select * from university_prod.hw_pack_run_avg_pwr limit 100

create table reporting.system_hw_pkg_power
(
    load_ts timestamp,
    guid text,
    dt date,
    instance int,
    nrs int,
    mean float,
    max float
);

alter table reporting.system_hw_pkg_power
    owner to postgres;

create index idx_hw_pkg_power_guid_dt on reporting.system_hw_pkg_power (guid, dt);

insert into reporting.system_hw_pkg_power
select
    load_ts,
    guid,
    dt,
    instance,
    nrs,
    mean,
    rap_22
from
    university_prod.hw_pack_run_avg_pwr;

select count(*) from reporting.system_hw_pkg_power;

select
    a.countryname_normalized,
    count(distinct b.guid) as number_of_systems,
    sum(nrs * mean)/sum(nrs) as avg_pkg_power_consumed
from
    reporting.system_sysinfo_unique_normalized a
inner join
    reporting.system_hw_pkg_power b
on
    a.guid = b.guid
group by
    a.countryname_normalized
order by avg_pkg_power_consumed desc;

select * from university_prod.power_acdc_usage_v4_hist;

select metric_name, count(*), count(distinct guid) from university_prod.power_acdc_usage_v4_hist group by 1;

select * from university_prod.power_acdc_usage_v4_hist where metric_name = 'HW::CORE:C0:PERCENT:'


select
    guid,
    dt,
    event_name,
    sum(nrs) as nrs,
    max(attribute_metric_level1) as number_of_cores,
    max(duration_ms) as duration_ms,
    sum(nrs*min_val)/sum(nrs) as min_pkg_C0,
    sum(nrs*avg_val)/sum(nrs) as avg_pkg_C0,
    sum(nrs*percentile_50th)/sum(nrs) as med_pkg_C0,
    sum(nrs*max_val)/sum(nrs) as max_pkg_C0
from
    university_prod.power_acdc_usage_v4_hist
where
    metric_name = 'HW::CORE:C0:PERCENT:'
group by
    guid,
    dt,
    event_name;

create table reporting.system_pkg_C0
(
    load_ts         timestamp default now(),
    guid            text,
    dt              date,
    event_name      text,
    nrs             bigint,
    number_of_cores smallint,
    duration_ms     bigint,
    min_pkg_c0      float,
    avg_pkg_c0      float,
    med_pkg_c0      float,
    max_pkg_c0      float
);

alter table reporting.system_pkg_C0
    owner to postgres;

create index idx_system_pkg_c0_guid_dt on reporting.system_pkg_C0 (guid, dt);

insert into reporting.system_pkg_C0
(guid, dt, event_name, nrs, number_of_cores, duration_ms, min_pkg_c0, avg_pkg_c0, med_pkg_c0, max_pkg_c0)
select
    guid,
    dt,
    event_name,
    sum(nrs) as nrs,
    max(attribute_metric_level1::smallint) as number_of_cores,
    max(duration_ms) as duration_ms,
    sum(nrs*min_val)/sum(nrs) as min_pkg_C0,
    sum(nrs*avg_val)/sum(nrs) as avg_pkg_C0,
    sum(nrs*percentile_50th)/sum(nrs) as med_pkg_C0,
    sum(nrs*max_val)/sum(nrs) as max_pkg_C0
from
    university_prod.power_acdc_usage_v4_hist
where
    metric_name = 'HW::CORE:C0:PERCENT:'
group by
    guid,
    dt,
    event_name;

select count(*) from reporting.system_pkg_C0;

select metric_name, count(distinct guid), count(*) from university_prod.power_acdc_usage_v4_hist group by metric_name;
select * from university_analysis_pad.data_dictionary_collector_inputs where input ilike '%epp%'
select * from university_analysis_pad.data_dictionary_inputs

select metric_name, count(distinct guid), count(*) from university_prod.power_acdc_usage_v4_hist group by metric_name;
select * from university_prod.power_acdc_usage_v4_hist where metric_name = 'HW::CORE:AVG_FREQ:MHZ:'

select
    guid,
    dt,
    event_name,
    max(nrs) as nrs,
    max(core) as number_of_cores,
    max(duration_ms) as duration_ms,
    avg(min_avg_freq_mhz) as min_avg_freq_mhz,
    avg(avg_avg_freq_mhz) as avg_avg_freq_mhz,
    avg(med_avg_freq_mhz) as med_avg_freq_mhz,
    avg(max_avg_freq_mhz) as max_avg_freq_mhz
from
(
    select
        guid,
        dt,
        event_name,
        attribute_metric_level1::smallint as core,
        sum(nrs) as nrs,
        sum(duration_ms) as duration_ms,
        sum(nrs*min_val)/sum(nrs) as min_avg_freq_mhz,
        sum(nrs*avg_val)/sum(nrs) as avg_avg_freq_mhz,
        sum(nrs*percentile_50th)/sum(nrs) as med_avg_freq_mhz,
        sum(nrs*max_val)/sum(nrs) as max_avg_freq_mhz
    from
        university_prod.power_acdc_usage_v4_hist
    where
        metric_name = 'HW::CORE:AVG_FREQ:MHZ:'
    group by
        guid,
        dt,
        event_name,
        attribute_metric_level1
) inn
group by
    guid,
    dt,
    event_name;

create table reporting.system_pkg_avg_freq_mhz
(
    load_ts timestamp default now(),
    guid text,
    dt date,
    event_name text,
    nrs bigint,
    number_of_cores smallint,
    duration_ms bigint,
    min_avg_freq_mhz float,
    avg_avg_freq_mhz float,
    med_avg_freq_mhz float,
    max_avg_freq_mhz float
);

alter table reporting.system_pkg_avg_freq_mhz
    owner to postgres;

create index idx_system_pkg_avg_freq_mhz_guid_dt on reporting.system_pkg_avg_freq_mhz (guid, dt);

insert into reporting.system_pkg_avg_freq_mhz
(guid, dt, event_name, nrs, number_of_cores, duration_ms, min_avg_freq_mhz, avg_avg_freq_mhz, med_avg_freq_mhz, max_avg_freq_mhz)
select
    guid,
    dt,
    event_name,
    max(nrs) as nrs,
    max(core) as number_of_cores,
    max(duration_ms) as duration_ms,
    avg(min_avg_freq_mhz) as min_avg_freq_mhz,
    avg(avg_avg_freq_mhz) as avg_avg_freq_mhz,
    avg(med_avg_freq_mhz) as med_avg_freq_mhz,
    avg(max_avg_freq_mhz) as max_avg_freq_mhz
from
(
    select
        guid,
        dt,
        event_name,
        attribute_metric_level1::smallint as core,
        sum(nrs) as nrs,
        sum(duration_ms) as duration_ms,
        sum(nrs*min_val)/sum(nrs) as min_avg_freq_mhz,
        sum(nrs*avg_val)/sum(nrs) as avg_avg_freq_mhz,
        sum(nrs*percentile_50th)/sum(nrs) as med_avg_freq_mhz,
        sum(nrs*max_val)/sum(nrs) as max_avg_freq_mhz
    from
        university_prod.power_acdc_usage_v4_hist
    where
        metric_name = 'HW::CORE:AVG_FREQ:MHZ:'
    group by
        guid,
        dt,
        event_name,
        attribute_metric_level1
) inn
group by
    guid,
    dt,
    event_name;

select * from reporting.system_pkg_avg_freq_mhz;

select metric_name, count(distinct guid), count(*) from university_prod.power_acdc_usage_v4_hist group by metric_name;
select * from university_prod.power_acdc_usage_v4_hist where metric_name = 'HW::CORE:TEMPERATURE:CENTIGRADE:'

select
    guid,
    dt,
    event_name,
    max(nrs) as nrs,
    max(core) as number_of_cores,
    max(duration_ms) as duration_ms,
    avg(min_temp_centigrade) as min_temp_centigrade,
    avg(avg_temp_centigrade) as avg_temp_centigrade,
    avg(med_temp_centigrade) as med_temp_centigrade,
    avg(max_temp_centigrade) as max_temp_centigrade
from
(
    select
        guid,
        dt,
        event_name,
        attribute_metric_level1::smallint as core,
        sum(nrs) as nrs,
        sum(duration_ms) as duration_ms,
        sum(nrs*min_val)/sum(nrs) as min_temp_centigrade,
        sum(nrs*avg_val)/sum(nrs) as avg_temp_centigrade,
        sum(nrs*percentile_50th)/sum(nrs) as med_temp_centigrade,
        sum(nrs*max_val)/sum(nrs) as max_temp_centigrade
    from
        university_prod.power_acdc_usage_v4_hist
    where
        metric_name = 'HW::CORE:TEMPERATURE:CENTIGRADE:'
    group by
        guid,
        dt,
        event_name,
        attribute_metric_level1
) inn
group by
    guid,
    dt,
    event_name;

create table reporting.system_pkg_temp_centigrade
(
    load_ts timestamp default now(),
    guid text,
    dt date,
    event_name text,
    nrs bigint,
    number_of_cores smallint,
    duration_ms bigint,
    min_temp_centigrade float,
    avg_temp_centigrade float,
    med_temp_centigrade float,
    max_temp_centigrade float
);

alter table reporting.system_pkg_avg_freq_mhz
    owner to postgres;

create index idx_system_pkg_temp_centigrade_guid_dt on reporting.system_pkg_temp_centigrade (guid, dt);

insert into reporting.system_pkg_temp_centigrade
(guid, dt, event_name, nrs, number_of_cores, duration_ms, min_temp_centigrade, avg_temp_centigrade, med_temp_centigrade, max_temp_centigrade)
select
    guid,
    dt,
    event_name,
    max(nrs) as nrs,
    max(core) as number_of_cores,
    max(duration_ms) as duration_ms,
    avg(min_temp_centigrade) as min_temp_centigrade,
    avg(avg_temp_centigrade) as avg_temp_centigrade,
    avg(med_temp_centigrade) as med_temp_centigrade,
    avg(max_temp_centigrade) as max_temp_centigrade
from
(
    select
        guid,
        dt,
        event_name,
        attribute_metric_level1::smallint as core,
        sum(nrs) as nrs,
        sum(duration_ms) as duration_ms,
        sum(nrs*min_val)/sum(nrs) as min_temp_centigrade,
        sum(nrs*avg_val)/sum(nrs) as avg_temp_centigrade,
        sum(nrs*percentile_50th)/sum(nrs) as med_temp_centigrade,
        sum(nrs*max_val)/sum(nrs) as max_temp_centigrade
    from
        university_prod.power_acdc_usage_v4_hist
    where
        metric_name = 'HW::CORE:TEMPERATURE:CENTIGRADE:'
    group by
        guid,
        dt,
        event_name,
        attribute_metric_level1
) inn
group by
    guid,
    dt,
    event_name;

select count(*) from reporting.system_pkg_temp_centigrade;
select * from reporting.system_pkg_temp_centigrade;

select metric_name, count(distinct guid), count(*) from university_prod.power_acdc_usage_v4_hist group by metric_name;
select * from university_prod.power_acdc_usage_v4_hist where metric_name = 'HW:::PSYS_RAP:WATTS:' and guid = '66f4917d6b9b48e69561f1235c3e6380' and dt = '2020-12-21'

select
    guid,
    dt,
    event_name,
    sum(nrs) as nrs,
    sum(duration_ms) as duration_ms,
    sum(nrs*min_val)/sum(nrs) as min_psys_rap_watts,
    sum(nrs*avg_val)/sum(nrs) as avg_psys_rap_watts,
    sum(nrs*percentile_50th)/sum(nrs) as med_psys_rap_watts,
    sum(nrs*max_val)/sum(nrs) as max_psys_rap_watts
from
    university_prod.power_acdc_usage_v4_hist
where
    metric_name = 'HW:::PSYS_RAP:WATTS:'
group by
    guid,
    dt,
    event_name;

create table reporting.system_psys_rap_watts
(
    load_ts timestamp default now(),
    guid text,
    dt date,
    event_name text,
    nrs bigint,
    duration_ms bigint,
    min_psys_rap_watts float,
    avg_psys_rap_watts float,
    med_psys_rap_watts float,
    max_psys_rap_watts float
);

alter table reporting.system_psys_rap_watts
    owner to postgres;

create index idx_system_psys_rap_watts_guid_dt on reporting.system_psys_rap_watts (guid, dt);

insert into reporting.system_psys_rap_watts
(guid, dt, event_name, nrs, duration_ms, min_psys_rap_watts, avg_psys_rap_watts, med_psys_rap_watts, max_psys_rap_watts)
select
    guid,
    dt,
    event_name,
    sum(nrs) as nrs,
    sum(duration_ms) as duration_ms,
    sum(nrs*min_val)/sum(nrs) as min_psys_rap_watts,
    sum(nrs*avg_val)/sum(nrs) as avg_psys_rap_watts,
    sum(nrs*percentile_50th)/sum(nrs) as med_psys_rap_watts,
    sum(nrs*max_val)/sum(nrs) as max_psys_rap_watts
from
    university_prod.power_acdc_usage_v4_hist
where
    metric_name = 'HW:::PSYS_RAP:WATTS:'
group by
    guid,
    dt,
    event_name;

select count(*) from reporting.system_psys_rap_watts;
select * from reporting.system_psys_rap_watts;

select * from university_analysis_pad.data_dictionary_collector_il;

select a.* from university_prod.web_cat_pivot a
inner join
    university_analysis_pad.system_sysinfo_unique_normalized b
on
    a.guid = b.guid;

create table reporting.system_web_cat_pivot
(
    guid                                       text,
    social_social_network                      double precision,
    private_private                            double precision,
    productivity_word_processing               double precision,
    news_news                                  double precision,
    social_communication                       double precision,
    productivity_spreadsheets                  double precision,
    content_creation_photo_edit_creation       double precision,
    search_search                              double precision,
    productivity_other                         double precision,
    entertainment_music_audio_streaming        double precision,
    finance_banking_and_accounting             double precision,
    games_other                                double precision,
    entertainment_other                        double precision,
    education_education                        double precision,
    productivity_programming                   double precision,
    productivity_presentations                 double precision,
    reference_reference                        double precision,
    shopping_shopping                          double precision,
    other_unclassified                         double precision,
    recreation_travel                          double precision,
    entertainment_video_streaming              double precision,
    games_video_games                          double precision,
    productivity_crm                           double precision,
    mail_mail                                  double precision,
    social_communication_live                  double precision,
    content_creation_video_audio_edit_creation double precision,
    productivity_project_management            double precision,
    content_creation_web_design_development    double precision
);

alter table reporting.system_web_cat_pivot
    owner to postgres;

create index idx_system_web_cat_pivot_guid on reporting.system_web_cat_pivot (guid);

insert into reporting.system_web_cat_pivot
(guid, social_social_network, private_private, productivity_word_processing, news_news, social_communication, productivity_spreadsheets, content_creation_photo_edit_creation, search_search, productivity_other, entertainment_music_audio_streaming, finance_banking_and_accounting, games_other, entertainment_other, education_education, productivity_programming, productivity_presentations, reference_reference, shopping_shopping, other_unclassified, recreation_travel, entertainment_video_streaming, games_video_games, productivity_crm, mail_mail, social_communication_live, content_creation_video_audio_edit_creation, productivity_project_management, content_creation_web_design_development)
select * from university_prod.web_cat_pivot;

select upper(network_interface), count(*), count(distinct guid) from university_prod.os_network_consumption_v2 group by 1;

select
    guid,
    dt,
    input_description,
    sum(nr_samples) as nrs,
    max(max_bytes_sec) as max_bytes_sec,
    min(min_bytes_sec) as min_bytes_sec,
    sum(nr_samples * avg_bytes_sec)/sum(nr_samples) as avg_bytes_sec
from
    university_prod.os_network_consumption_v2
group by
    guid, dt, input_description;

create table reporting.system_network_consumption
(
    load_ts timestamp default now(),
    guid text,
    dt date,
    input_desc text,
    nrs int,
    min_bytes_sec float,
    avg_bytes_sec float,
    max_bytes_sec float
);

alter table reporting.system_network_consumption
    owner to postgres;

create index idx_system_network_consumption_guid on reporting.system_network_consumption (guid);

truncate table reporting.system_network_consumption;
insert into reporting.system_network_consumption
(guid, dt, input_desc, nrs, min_bytes_sec, avg_bytes_sec, max_bytes_sec)
select
    guid,
    dt,
    input_description,
    sum(nr_samples) as nrs,
    min(min_bytes_sec) as min_bytes_sec,
    sum(nr_samples * avg_bytes_sec)/sum(nr_samples) as avg_bytes_sec,
    max(max_bytes_sec) as max_bytes_sec
from
    university_prod.os_network_consumption_v2
group by
    guid, dt, input_description;

select guid, nrs, received_bytes_sec, sent_bytes_sec
from
(
    select
        guid,
        sum(nrs) as nrs,
        sum(case
             when input_desc = 'OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::' then avg_bytes_sec
             else 0 end) as received_bytes_sec,
        sum(case
             when input_desc = 'OS:NETWORK INTERFACE::BYTES SENT/SEC::' then avg_bytes_sec
             else 0 end) as sent_bytes_sec
    from
        reporting.system_network_consumption
    group by
        guid
    having
        sum(nrs) > 720
) a
where
    sent_bytes_sec > received_bytes_sec;

select count(distinct guid) from reporting.system_network_consumption;

select * from reporting.system_network_consumption where guid = '01560155af18493cafceb38147047de5' --and dt = '2022-08-18';

select * from university_prod.os_network_consumption_v2 where guid = '01560155af18493cafceb38147047de5' --and dt = '2022-08-18';

select
    a.guid,
    nrs,
    received_bytes,
    sent_bytes,
    b.chassistype,
    b.modelvendor_normalized as vendor,
    b.model_normalized as model,
    b.ram,
    b.os,
    b."#ofcores" as number_of_cores,
    b.processor_line,
    b.processornumber,
    b.cpucode,
    b.cpuname
from
(
    select
        guid,
        sum(nrs) as nrs,
        sum(case
             when input_desc = 'OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::' then avg_bytes_sec * nrs * 5
             else 0 end) as received_bytes,
        sum(case
             when input_desc = 'OS:NETWORK INTERFACE::BYTES SENT/SEC::' then avg_bytes_sec * nrs * 5
             else 0 end) as sent_bytes
    from
        reporting.system_network_consumption
    group by
        guid
    having
        sum(nrs) > 720
) a
inner join
    reporting.system_sysinfo_unique_normalized b
on
    a.guid = b.guid
where
    sent_bytes > received_bytes;

select cpuname, cpucode, count(*) from reporting.system_sysinfo_unique_normalized where cpuname = 'Xeon' group by 1, 2


select
    processor_class,
    os,
    count(distinct guid) as number_of_systems,
    sum(nrs * avg_bytes_received)/sum(nrs) as avg_bytes_received,
    sum(nrs * avg_bytes_sent)/sum(nrs) as avg_bytes_sent
from (select a.guid,
             a.cpuname,
             a.cpucode,
             a.os,
             sum(b.nrs) as nrs,
             case when cpuname = 'Xeon' then 'Server Class' else 'Non-Server Class' end as processor_class,
             sum(case
                     when b.input_desc = 'OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::' then b.avg_bytes_sec * b.nrs * 5
                     else 0 end)                                                        as avg_bytes_received,
             sum(case
                     when b.input_desc = 'OS:NETWORK INTERFACE::BYTES SENT/SEC::' then b.avg_bytes_sec * b.nrs * 5
                     else 0 end)                                                        as avg_bytes_sent
      from reporting.system_sysinfo_unique_normalized a
               inner join
           reporting.system_network_consumption b
           on
               a.guid = b.guid
      group by a.guid, a.cpuname, a.cpucode, a.os) c
group by
    processor_class, os;

select count(distinct a.guid) from reporting.system_sysinfo_unique_normalized a
inner join
    reporting.system_network_consumption b
on
    a.guid = b.guid
where os = 'Win Server'

select * from university_prod.os_system_data limit 100;

select * from university_prod.os_system_gen_data limit 100;

select * from university_prod.web_cat_usage_v2 limit 100;

select count(distinct guid) from reporting.system_web_cat_pivot limit 100;

select * from reporting.system_web_cat_pivot;

select parent_category || '-' || sub_category, count(*) from university_prod.web_cat_usage_v2 group by parent_category, sub_category

select guid, sum(duration_ms) from university_prod.web_cat_usage_v2 where guid='00075517414d434fb03f4e8027f0ab61' and parent_category='social' and sub_category='communication - live' group by guid;

select * from reporting.system_web_cat_pivot where guid = '00075517414d434fb03f4e8027f0ab61'

select * from university_prod.web_cat_usage_v2 order by random();

select

select
    browser,
    round(num_systems *100.0 / total_systems, 2) as percent_systems,
    round(num_instances *100.0 / tot_instances, 2) as percent_instances,
    round(sum_duration *100.0 / tot_duration, 2) as percent_duration
from (select browser, count(distinct guid) as num_systems, count(*) as num_instances, sum(duration_ms) as sum_duration
      from university_prod.web_cat_usage_v2
      group by browser) a
cross join
     (select count(distinct guid) as total_systems, count(*) as tot_instances, sum(duration_ms) as tot_duration
      from university_prod.web_cat_usage_v2) b;

select guid||dt, count(*) from university_prod.web_cat_usage_v2 group by guid||dt

select * from university_prod.web_cat_usage_v2 where guid = '000091c0adc149389235ed2c5f15a59e' and dt = '2021-06-11'

select count(*) from university_prod.web_cat_usage_v2;


select
    guid,
    dt,
    sum(case when parent_category = 'content creation' and sub_category = 'photo edit/creation' then duration_ms else 0 end ) as content_creation_photo_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'video/audio edit/creation' then duration_ms else 0 end) as content_creation_video_audio_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'web design / development' then duration_ms else 0 end) as content_creation_web_design_development,
    sum(case when parent_category = 'education' then duration_ms else 0 end) as education,
    sum(case when parent_category = 'entertainment' and sub_category = 'music / audio streaming' then duration_ms else 0 end) as entertainment_music_audio_streaming,
    sum(case when parent_category = 'entertainment' and sub_category = 'other' then duration_ms else 0 end) as entertainment_other,
    sum(case when parent_category = 'entertainment' and sub_category = 'video streaming' then duration_ms else 0 end) as entertainment_video_streaming,
    sum(case when parent_category = 'finance' then duration_ms else 0 end) as finance,
    sum(case when parent_category = 'games' and sub_category = 'other' then duration_ms else 0 end) as games_other,
    sum(case when parent_category = 'games' and sub_category = 'video games' then duration_ms else 0 end) as games_video_games,
    sum(case when parent_category = 'mail' then duration_ms else 0 end) as mail,
    sum(case when parent_category = 'news' then duration_ms else 0 end) as news,
    sum(case when parent_category = 'other' then duration_ms else 0 end) as unclassified,
    sum(case when parent_category = 'private' then duration_ms else 0 end) as private,
    sum(case when parent_category = 'productivity' and sub_category = 'crm' then duration_ms else 0 end) as productivity_crm,
    sum(case when parent_category = 'productivity' and sub_category = 'other' then duration_ms else 0 end) as productivity_other,
    sum(case when parent_category = 'productivity' and sub_category = 'presentations' then duration_ms else 0 end) as productivity_presentations,
    sum(case when parent_category = 'productivity' and sub_category = 'programming' then duration_ms else 0 end) as productivity_programming,
    sum(case when parent_category = 'productivity' and sub_category = 'project management' then duration_ms else 0 end) as productivity_project_management,
    sum(case when parent_category = 'productivity' and sub_category = 'spreadsheets' then duration_ms else 0 end) as productivity_spreadsheets,
    sum(case when parent_category = 'productivity' and sub_category = 'word processing' then duration_ms else 0 end) as productivity_word_processing,
    sum(case when parent_category = 'recreation' and sub_category = 'travel' then duration_ms else 0 end) as recreation_travel,
    sum(case when parent_category = 'reference' then duration_ms else 0 end) as reference,
    sum(case when parent_category = 'search' then duration_ms else 0 end) as search,
    sum(case when parent_category = 'shopping' then duration_ms else 0 end) as shopping,
    sum(case when parent_category = 'social' and sub_category = 'social network' then duration_ms else 0 end) as social_social_network,
    sum(case when parent_category = 'social' and sub_category = 'communication' then duration_ms else 0 end) as social_communication,
    sum(case when parent_category = 'social' and sub_category = 'communication - live' then duration_ms else 0 end) as social_communication_live
into reporting.system_web_cat_pivot_duration
from
    university_prod.web_cat_usage_v2
group by 1, 2;

create index idx_system_web_cat_pivot_duration_guid_dt on reporting.system_web_cat_pivot_duration (guid, dt);

select
    guid,
    dt,
    sum(case when parent_category = 'content creation' and sub_category = 'photo edit/creation' then page_load_count else 0 end ) as content_creation_photo_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'video/audio edit/creation' then page_load_count else 0 end) as content_creation_video_audio_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'web design / development' then page_load_count else 0 end) as content_creation_web_design_development,
    sum(case when parent_category = 'education' then page_load_count else 0 end) as education,
    sum(case when parent_category = 'entertainment' and sub_category = 'music / audio streaming' then page_load_count else 0 end) as entertainment_music_audio_streaming,
    sum(case when parent_category = 'entertainment' and sub_category = 'other' then page_load_count else 0 end) as entertainment_other,
    sum(case when parent_category = 'entertainment' and sub_category = 'video streaming' then page_load_count else 0 end) as entertainment_video_streaming,
    sum(case when parent_category = 'finance' then page_load_count else 0 end) as finance,
    sum(case when parent_category = 'games' and sub_category = 'other' then page_load_count else 0 end) as games_other,
    sum(case when parent_category = 'games' and sub_category = 'video games' then page_load_count else 0 end) as games_video_games,
    sum(case when parent_category = 'mail' then page_load_count else 0 end) as mail,
    sum(case when parent_category = 'news' then page_load_count else 0 end) as news,
    sum(case when parent_category = 'other' then page_load_count else 0 end) as unclassified,
    sum(case when parent_category = 'private' then page_load_count else 0 end) as private,
    sum(case when parent_category = 'productivity' and sub_category = 'crm' then page_load_count else 0 end) as productivity_crm,
    sum(case when parent_category = 'productivity' and sub_category = 'other' then page_load_count else 0 end) as productivity_other,
    sum(case when parent_category = 'productivity' and sub_category = 'presentations' then page_load_count else 0 end) as productivity_presentations,
    sum(case when parent_category = 'productivity' and sub_category = 'programming' then page_load_count else 0 end) as productivity_programming,
    sum(case when parent_category = 'productivity' and sub_category = 'project management' then page_load_count else 0 end) as productivity_project_management,
    sum(case when parent_category = 'productivity' and sub_category = 'spreadsheets' then page_load_count else 0 end) as productivity_spreadsheets,
    sum(case when parent_category = 'productivity' and sub_category = 'word processing' then page_load_count else 0 end) as productivity_word_processing,
    sum(case when parent_category = 'recreation' and sub_category = 'travel' then page_load_count else 0 end) as recreation_travel,
    sum(case when parent_category = 'reference' then page_load_count else 0 end) as reference,
    sum(case when parent_category = 'search' then page_load_count else 0 end) as search,
    sum(case when parent_category = 'shopping' then page_load_count else 0 end) as shopping,
    sum(case when parent_category = 'social' and sub_category = 'social network' then page_load_count else 0 end) as social_social_network,
    sum(case when parent_category = 'social' and sub_category = 'communication' then page_load_count else 0 end) as social_communication,
    sum(case when parent_category = 'social' and sub_category = 'communication - live' then page_load_count else 0 end) as social_communication_live
into reporting.system_web_cat_pivot_page_load_count
from
    university_prod.web_cat_usage_v2
group by 1, 2;

create index idx_system_web_cat_pivot_page_load_count_guid_dt on reporting.system_web_cat_pivot_page_load_count (guid, dt);

select
    guid,
    dt,
    sum(case when parent_category = 'content creation' and sub_category = 'photo edit/creation' then page_visit_count else 0 end ) as content_creation_photo_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'video/audio edit/creation' then page_visit_count else 0 end) as content_creation_video_audio_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'web design / development' then page_visit_count else 0 end) as content_creation_web_design_development,
    sum(case when parent_category = 'education' then page_visit_count else 0 end) as education,
    sum(case when parent_category = 'entertainment' and sub_category = 'music / audio streaming' then page_visit_count else 0 end) as entertainment_music_audio_streaming,
    sum(case when parent_category = 'entertainment' and sub_category = 'other' then page_visit_count else 0 end) as entertainment_other,
    sum(case when parent_category = 'entertainment' and sub_category = 'video streaming' then page_visit_count else 0 end) as entertainment_video_streaming,
    sum(case when parent_category = 'finance' then page_visit_count else 0 end) as finance,
    sum(case when parent_category = 'games' and sub_category = 'other' then page_visit_count else 0 end) as games_other,
    sum(case when parent_category = 'games' and sub_category = 'video games' then page_visit_count else 0 end) as games_video_games,
    sum(case when parent_category = 'mail' then page_visit_count else 0 end) as mail,
    sum(case when parent_category = 'news' then page_visit_count else 0 end) as news,
    sum(case when parent_category = 'other' then page_visit_count else 0 end) as unclassified,
    sum(case when parent_category = 'private' then page_visit_count else 0 end) as private,
    sum(case when parent_category = 'productivity' and sub_category = 'crm' then page_visit_count else 0 end) as productivity_crm,
    sum(case when parent_category = 'productivity' and sub_category = 'other' then page_visit_count else 0 end) as productivity_other,
    sum(case when parent_category = 'productivity' and sub_category = 'presentations' then page_visit_count else 0 end) as productivity_presentations,
    sum(case when parent_category = 'productivity' and sub_category = 'programming' then page_visit_count else 0 end) as productivity_programming,
    sum(case when parent_category = 'productivity' and sub_category = 'project management' then page_visit_count else 0 end) as productivity_project_management,
    sum(case when parent_category = 'productivity' and sub_category = 'spreadsheets' then page_visit_count else 0 end) as productivity_spreadsheets,
    sum(case when parent_category = 'productivity' and sub_category = 'word processing' then page_visit_count else 0 end) as productivity_word_processing,
    sum(case when parent_category = 'recreation' and sub_category = 'travel' then page_visit_count else 0 end) as recreation_travel,
    sum(case when parent_category = 'reference' then page_visit_count else 0 end) as reference,
    sum(case when parent_category = 'search' then page_visit_count else 0 end) as search,
    sum(case when parent_category = 'shopping' then page_visit_count else 0 end) as shopping,
    sum(case when parent_category = 'social' and sub_category = 'social network' then page_visit_count else 0 end) as social_social_network,
    sum(case when parent_category = 'social' and sub_category = 'communication' then page_visit_count else 0 end) as social_communication,
    sum(case when parent_category = 'social' and sub_category = 'communication - live' then page_visit_count else 0 end) as social_communication_live
into reporting.system_web_cat_pivot_page_visit_count
from
    university_prod.web_cat_usage_v2
group by 1, 2;

create index idx_system_web_cat_pivot_page_visit_count_guid_dt on reporting.system_web_cat_pivot_page_visit_count (guid, dt);


select
    guid,
    dt,
    sum(case when parent_category = 'content creation' and sub_category = 'photo edit/creation' then domain_count else 0 end ) as content_creation_photo_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'video/audio edit/creation' then domain_count else 0 end) as content_creation_video_audio_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'web design / development' then domain_count else 0 end) as content_creation_web_design_development,
    sum(case when parent_category = 'education' then domain_count else 0 end) as education,
    sum(case when parent_category = 'entertainment' and sub_category = 'music / audio streaming' then domain_count else 0 end) as entertainment_music_audio_streaming,
    sum(case when parent_category = 'entertainment' and sub_category = 'other' then domain_count else 0 end) as entertainment_other,
    sum(case when parent_category = 'entertainment' and sub_category = 'video streaming' then domain_count else 0 end) as entertainment_video_streaming,
    sum(case when parent_category = 'finance' then domain_count else 0 end) as finance,
    sum(case when parent_category = 'games' and sub_category = 'other' then domain_count else 0 end) as games_other,
    sum(case when parent_category = 'games' and sub_category = 'video games' then domain_count else 0 end) as games_video_games,
    sum(case when parent_category = 'mail' then domain_count else 0 end) as mail,
    sum(case when parent_category = 'news' then domain_count else 0 end) as news,
    sum(case when parent_category = 'other' then domain_count else 0 end) as unclassified,
    sum(case when parent_category = 'private' then domain_count else 0 end) as private,
    sum(case when parent_category = 'productivity' and sub_category = 'crm' then domain_count else 0 end) as productivity_crm,
    sum(case when parent_category = 'productivity' and sub_category = 'other' then domain_count else 0 end) as productivity_other,
    sum(case when parent_category = 'productivity' and sub_category = 'presentations' then domain_count else 0 end) as productivity_presentations,
    sum(case when parent_category = 'productivity' and sub_category = 'programming' then domain_count else 0 end) as productivity_programming,
    sum(case when parent_category = 'productivity' and sub_category = 'project management' then domain_count else 0 end) as productivity_project_management,
    sum(case when parent_category = 'productivity' and sub_category = 'spreadsheets' then domain_count else 0 end) as productivity_spreadsheets,
    sum(case when parent_category = 'productivity' and sub_category = 'word processing' then domain_count else 0 end) as productivity_word_processing,
    sum(case when parent_category = 'recreation' and sub_category = 'travel' then domain_count else 0 end) as recreation_travel,
    sum(case when parent_category = 'reference' then domain_count else 0 end) as reference,
    sum(case when parent_category = 'search' then domain_count else 0 end) as search,
    sum(case when parent_category = 'shopping' then domain_count else 0 end) as shopping,
    sum(case when parent_category = 'social' and sub_category = 'social network' then domain_count else 0 end) as social_social_network,
    sum(case when parent_category = 'social' and sub_category = 'communication' then domain_count else 0 end) as social_communication,
    sum(case when parent_category = 'social' and sub_category = 'communication - live' then domain_count else 0 end) as social_communication_live
into reporting.system_web_cat_pivot_domain_count
from
    university_prod.web_cat_usage_v2
group by 1, 2;

create index idx_system_web_cat_pivot_domain_count_guid_dt on reporting.system_web_cat_pivot_domain_count (guid, dt);

select
    guid,
    dt,
    sum(case when parent_category = 'content creation' and sub_category = 'photo edit/creation' then page_visit_count else 0 end ) as content_creation_photo_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'video/audio edit/creation' then page_visit_count else 0 end) as content_creation_video_audio_edit_creation,
    sum(case when parent_category = 'content creation' and sub_category = 'web design / development' then page_visit_count else 0 end) as content_creation_web_design_development,
    sum(case when parent_category = 'education' then page_visit_count else 0 end) as education,
    sum(case when parent_category = 'entertainment' and sub_category = 'music / audio streaming' then page_visit_count else 0 end) as entertainment_music_audio_streaming,
    sum(case when parent_category = 'entertainment' and sub_category = 'other' then page_visit_count else 0 end) as entertainment_other,
    sum(case when parent_category = 'entertainment' and sub_category = 'video streaming' then page_visit_count else 0 end) as entertainment_video_streaming,
    sum(case when parent_category = 'finance' then page_visit_count else 0 end) as finance,
    sum(case when parent_category = 'games' and sub_category = 'other' then page_visit_count else 0 end) as games_other,
    sum(case when parent_category = 'games' and sub_category = 'video games' then page_visit_count else 0 end) as games_video_games,
    sum(case when parent_category = 'mail' then page_visit_count else 0 end) as mail,
    sum(case when parent_category = 'news' then page_visit_count else 0 end) as news,
    sum(case when parent_category = 'other' then page_visit_count else 0 end) as unclassified,
    sum(case when parent_category = 'private' then page_visit_count else 0 end) as private,
    sum(case when parent_category = 'productivity' and sub_category = 'crm' then page_visit_count else 0 end) as productivity_crm,
    sum(case when parent_category = 'productivity' and sub_category = 'other' then page_visit_count else 0 end) as productivity_other,
    sum(case when parent_category = 'productivity' and sub_category = 'presentations' then page_visit_count else 0 end) as productivity_presentations,
    sum(case when parent_category = 'productivity' and sub_category = 'programming' then page_visit_count else 0 end) as productivity_programming,
    sum(case when parent_category = 'productivity' and sub_category = 'project management' then page_visit_count else 0 end) as productivity_project_management,
    sum(case when parent_category = 'productivity' and sub_category = 'spreadsheets' then page_visit_count else 0 end) as productivity_spreadsheets,
    sum(case when parent_category = 'productivity' and sub_category = 'word processing' then page_visit_count else 0 end) as productivity_word_processing,
    sum(case when parent_category = 'recreation' and sub_category = 'travel' then page_visit_count else 0 end) as recreation_travel,
    sum(case when parent_category = 'reference' then page_visit_count else 0 end) as reference,
    sum(case when parent_category = 'search' then page_visit_count else 0 end) as search,
    sum(case when parent_category = 'shopping' then page_visit_count else 0 end) as shopping,
    sum(case when parent_category = 'social' and sub_category = 'social network' then page_visit_count else 0 end) as social_social_network,
    sum(case when parent_category = 'social' and sub_category = 'communication' then page_visit_count else 0 end) as social_communication,
    sum(case when parent_category = 'social' and sub_category = 'communication - live' then page_visit_count else 0 end) as social_communication_live
into reporting.system_web_cat_pivot_page_visit_count
from
    university_prod.web_cat_usage_v2
group by 1, 2;

create index idx_system_web_cat_pivot_page_visit_count_guid_dt on reporting.system_web_cat_pivot_page_visit_count (guid, dt);


select distinct parent_category || '-' || sub_category as category from university_prod.web_cat_usage_v2 order by parent_category || '-' || sub_category;

select count(*) from reporting.system_web_cat_pivot_page_load_count;
select * from reporting.system_web_cat_pivot_page_load_count;

select * from university_prod.web_cat_usage_v2;

select
    browser,
    round(num_systems *100.0 / total_systems, 2) as percent_systems,
    round(num_instances *100.0 / tot_instances, 2) as percent_instances,
    round(sum_duration *100.0 / tot_duration, 2) as percent_duration
from (select browser, count(distinct guid) as num_systems, count(*) as num_instances, sum(duration_ms) as sum_duration
      from university_prod.web_cat_usage_v2
      group by browser) a
cross join
     (select count(distinct guid) as total_systems, count(*) as tot_instances, sum(duration_ms) as tot_duration
      from university_prod.web_cat_usage_v2) b;

select
    guid,
    dt,
    browser,
    parent_category,
    sub_category,
    sum(duration_ms) as duration_ms,
    sum(page_load_count) as page_load_count,
    sum(site_count) as site_count,
    sum(domain_count) as domain_count,
    sum(page_visit_count) as page_visit_count
from
    university_prod.web_cat_usage_v2
group by
    guid, dt, browser, parent_category, sub_category;

create table reporting.system_web_cat_usage
(
    load_ts timestamp default now(),
    guid text,
    dt date,
    browser text,
    parent_category text,
    sub_category text,
    duration_ms bigint,
    page_load_count int,
    site_count int,
    domain_count int,
    page_visit_count int
);

alter table reporting.system_web_cat_usage
    owner to postgres;

create index idx_system_web_cat_usage_guid_dt on reporting.system_web_cat_usage (guid, dt);

insert into reporting.system_web_cat_usage
(guid, dt, browser, parent_category, sub_category, duration_ms, page_load_count, site_count, domain_count, page_visit_count)
select
    guid,
    dt,
    browser,
    parent_category,
    sub_category,
    sum(duration_ms) as duration_ms,
    sum(page_load_count) as page_load_count,
    sum(site_count) as site_count,
    sum(domain_count) as domain_count,
    sum(page_visit_count) as page_visit_count
from
    university_prod.web_cat_usage_v2
group by
    guid, dt, browser, parent_category, sub_category;

select
    browser,
    round(num_systems *100.0 / total_systems, 2) as percent_systems,
    round(num_instances *100.0 / tot_instances, 2) as percent_instances,
    round(sum_duration *100.0 / tot_duration, 2) as percent_duration
from (select browser, count(distinct guid) as num_systems, count(*) as num_instances, sum(duration_ms) as sum_duration
      from reporting.system_web_cat_usage
      group by browser) a
cross join
     (select count(distinct guid) as total_systems, count(*) as tot_instances, sum(duration_ms) as tot_duration
      from reporting.system_web_cat_usage) b;


select
    country,
    browser
from (select countryname_normalized                                                                 as country,
             browser,
             count(distinct b.guid)                                                                 as number_of_systems,
             rank() over (partition by countryname_normalized order by count(distinct b.guid) desc) as rnk
      from reporting.system_sysinfo_unique_normalized a
               inner join
           reporting.system_web_cat_usage b
           on
               a.guid = b.guid
      group by countryname_normalized, browser
      order by countryname_normalized, number_of_systems desc)
where rnk = 1;

select * from university_prod.os_memsam_avail_percent;

select guid, ram * 2^10 as ram_mb from reporting.system_sysinfo_unique_normalized;

select guid, interval_local_start::date as dt, (value::bigint) / 2^20 from university_prod.os_system_gen_data where input_desc = 'OS:QUERY::MEM_CHIPS::' and key='Capacity';

select * from university_analysis_pad.data_dictionary where table_name = 'os_memsam_avail_percent';

select
    guid,
    dt,
    nrs,
    avg_free_memory as avg_free_ram,
    sysinfo_ram,
    sysinfo_ram - avg_free_memory as utilized_ram,
    round((sysinfo_ram - avg_free_memory)*100/sysinfo_ram) as average_percentage_used
from (select a.guid,
             a.dt,
             sum(a.sample_count)                                   as nrs,
             sum(a.sample_count * a.average) / sum(a.sample_count) as avg_free_memory,
             b.ram * 2 ^ 10                                        as sysinfo_ram
      from university_prod.os_memsam_avail_percent a
               inner join
           reporting.system_sysinfo_unique_normalized b
           on
               a.guid = b.guid
      where
          b.ram != 0
      group by a.guid, a.dt, b.ram) c;


create table reporting.system_memory_utilization
(
    load_ts timestamp default now(),
    guid text,
    dt date,
    nrs int,
    avg_free_ram float,
    sysinfo_ram int,
    utilized_ram float,
    avg_percentage_used float
);

alter table reporting.system_memory_utilization
    owner to postgres;

create index idx_system_memory_utilization on reporting.system_memory_utilization (guid, dt);

insert into reporting.system_memory_utilization
(guid, dt, nrs, avg_free_ram, sysinfo_ram, utilized_ram, avg_percentage_used)
select
    guid,
    dt,
    nrs,
    avg_free_memory as avg_free_ram,
    sysinfo_ram,
    sysinfo_ram - avg_free_memory as utilized_ram,
    round((sysinfo_ram - avg_free_memory)*100/sysinfo_ram) as average_percentage_used
from (select a.guid,
             a.dt,
             sum(a.sample_count)                                   as nrs,
             sum(a.sample_count * a.average) / sum(a.sample_count) as avg_free_memory,
             b.ram * 2 ^ 10                                        as sysinfo_ram
      from university_prod.os_memsam_avail_percent a
               inner join
           reporting.system_sysinfo_unique_normalized b
           onselect
    sysinfo_ram,
    count(distinct guid),
    sum(nrs*average_percentage_used)/sum(nrs) as avg_percentage_used
from
    reporting.system_memory_utilization
group by sysinfo_ram
order by sysinfo_ram asc
               a.guid = b.guid
      where
          b.ram != 0
      group by a.guid, a.dt, b.ram) c;

select count(distinct guid) from reporting.system_memory_utilization where utilized_ram < 0;

select
    sysinfo_ram/2^10 as ram_gb,
    count(distinct guid),
    round(sum(nrs*avg_percentage_used)/sum(nrs)) as avg_percentage_used
from
    reporting.system_memory_utilization
where avg_percentage_used > 0
group by sysinfo_ram
order by sysinfo_ram asc;

select * from university_prod.os_system_data where key = 'Status' and value  = 'Pred Fail';

select * from university_prod.os_system_data where guid = '03af26d6554e4842ab11d2a802db5f3d' and input_desc ilike 'DISK%'

select count(distinct guid), count(*) from university_analysis_pad.mods_sleepstudy_power_estimation_data_13wks;

select app_id,  regexp_replace(app_id, '^.*\\', '') from university_analysis_pad.mods_sleepstudy_power_estimation_data_13wks;

SELECT regexp_replace('\Device\HarddiskVolume3\Program Files (x86)\McAfee Security Scan\4.1.583\mc-webview-cnt.exe', '^.*\\', '')


select * from university_analysis_pad.mods_sleepstudy_power_estimation_data_13wks;

select distinct user_id from university_analysis_pad.mods_sleepstudy_power_estimation_data_13wks;


select
    guid,
    dt,
    app_id,
    user_id,
    sum(cpu_power_consumption) as cpu_power_consumption,
    sum(display_power_consumption) as display_power_consumption,
    sum(disk_power_consumption) as disk_power_consumption,
    sum(mbb_power_consumption) as mbb_power_consumption,
    sum(network_power_consumption) as network_power_consumption,
    sum(soc_power_consumption) as soc_power_consumption,
    sum(loss_power_consumption) as loss_power_consumption,
    sum(other_power_consumption) as other_power_consumption,
    sum(total_power_consumption) as total_power_consumption
from (select guid,
             dt_utc,
             ts_local,
             ts_local::date                      as dt,
             regexp_replace(app_id, '^.*\\', '') as app_id,
             user_id,
             cpu_power_consumption,
             display_power_consumption,
             disk_power_consumption,
             mbb_power_consumption,
             network_power_consumption,
             soc_power_consumption,
             loss_power_consumption,
             other_power_consumption,
             total_power_consumption
      from university_analysis_pad.mods_sleepstudy_power_estimation_data_13wks) a
group by
    guid, dt, app_id, user_id;

create table reporting.system_mods_power_consumption
(
    load_ts timestamp default now(),
    guid text,
    dt date,
    app_id text,
    user_id text,
    cpu_power_consumption int,
    display_power_consumption int,
    disk_power_consumption int,
    mbb_power_consumption int,
    network_power_consumption int,
    soc_power_consumption int,
    loss_power_consumption int,
    other_power_consumption int,
    total_power_consumption int
);

alter table reporting.system_mods_power_consumption
    owner to postgres;

create index idx_system_mods_power_consumption_guid_dt_app_id on reporting.system_mods_power_consumption (guid, dt, app_id);

insert into reporting.system_mods_power_consumption
(guid, dt, app_id, user_id, cpu_power_consumption, display_power_consumption, disk_power_consumption, mbb_power_consumption, network_power_consumption, soc_power_consumption, loss_power_consumption, other_power_consumption, total_power_consumption)
select
    guid,
    dt,
    app_id,
    user_id,
    sum(cpu_power_consumption) as cpu_power_consumption,
    sum(display_power_consumption) as display_power_consumption,
    sum(disk_power_consumption) as disk_power_consumption,
    sum(mbb_power_consumption) as mbb_power_consumption,
    sum(network_power_consumption) as network_power_consumption,
    sum(soc_power_consumption) as soc_power_consumption,
    sum(loss_power_consumption) as loss_power_consumption,
    sum(other_power_consumption) as other_power_consumption,
    sum(total_power_consumption) as total_power_consumption
from (select guid,
             dt_utc,
             ts_local,
             ts_local::date                      as dt,
             regexp_replace(app_id, '^.*\\', '') as app_id,
             user_id,
             cpu_power_consumption,
             display_power_consumption,
             disk_power_consumption,
             mbb_power_consumption,
             network_power_consumption,
             soc_power_consumption,
             loss_power_consumption,
             other_power_consumption,
             total_power_consumption
      from university_analysis_pad.mods_sleepstudy_power_estimation_data_13wks) a
group by
    guid, dt, app_id, user_id;

select count(*) from reporting.system_mods_power_consumption;

select
    user_id,
    sum(total_power_consumption) as total_power_consumption,
    rank() over (order by sum(total_power_consumption) desc )as rnk
from
    reporting.system_mods_power_consumption
group by user_id;

select
    user_id,
    app_id,
    total_power_consumption,
    rnk
from (select user_id,
             app_id,
             sum(total_power_consumption)                                                   as total_power_consumption,
             rank() over (partition by user_id order by sum(total_power_consumption) desc ) as rnk
      from reporting.system_mods_power_consumption
      group by user_id, app_id)
where rnk <= 10;

select * from university_analysis_pad.mods_sleepstudy_scenario_instance_13wks order by audit_internal_path, ts_utc;

select guid, ts_utc, type, count(*) from university_analysis_pad.mods_sleepstudy_scenario_instance_13wks group by 1, 2, 3;


CsExitTimeSpentInPlm
ScreenOnOverhead
MonitorPowerOnTime
IsDebuggerEnabled
CsExitTimeSpentInDam
CsExitLatencyFromResiliencyExitToPLMEntryTime
PreVetoCount
Discharge
DwmSyncFlushTime
CsExitLatencyFromResiliencyExitToLowPowerEpochEntryTime
CsExitTimeSpentInResiliency
Id
EnergySaverPolicy
CsExitLatencyInMs
EntryFullCapacity
BatteryCountChanged
CsExitTimeSpentInResiliencyNotif
CsExitTimeSpentInPresence
LocalTimestamp
EntryRemainingCapacity
DirectedDripsTransitionCount
ExitRemainingCapacity
CsExitTimeSpentInLowPower
ActivityLevel
CsExitLatencyFromResiliencyExitToDAMEntryTime
LowPowerStateTime
OsStateId
EnergyDrain
IsHibernateEnabled
CsEnterReason
HwLowPowerStateTime
CsExitTimeSpentInMaintenance
CsExitTimeSpentInScreenOn
SpmScenarioStopReason
ExitFullCapacity
HibernateTimeoutInSec
FullChargeCapacity
HibernateBudgetPercentage
CsExitTimeSpentInConnection
DisconnectedStandby
Ac
CsExitReason
GdiOnTime
SystemGpuAdapterCount

select
    a.chassistype,
    count(distinct a.guid) as number_of_systems,
    sum(b.nrs * b.avg_psys_rap_watts)/sum(b.nrs) as avg_psys_rap_watts,
    sum(c.nrs * c.avg_pkg_c0)/sum(c.nrs) as avg_pkg_c0,
    sum(d.nrs * d.avg_avg_freq_mhz)/sum(d.nrs) as avg_freq_mhz,
    sum(e.nrs * e.avg_temp_centigrade)/sum(e.nrs) as avg_temp_centigrade
from
    reporting.system_sysinfo_unique_normalized a
inner join
    reporting.system_psys_rap_watts b
on
    a.guid = b.guid
inner join
    reporting.system_pkg_C0 c
on
    a.guid = c.guid
inner join
    reporting.system_pkg_avg_freq_mhz d
on
    a.guid = d.guid
inner join
    reporting.system_pkg_temp_centigrade e
on
    a.guid = e.guid
group by
    a.chassistype;

select
    app_id,
    total_power_consumption,
    rnk
from (select
             app_id,
             avg(total_power_consumption)                                                   as total_power_consumption,
             rank() over (order by sum(total_power_consumption) desc ) as rnk
      from reporting.system_mods_power_consumption
      group by app_id)
where rnk <= 20;


select
    persona,
    count(distinct guid) as number_of_systems,
    sum(days) as days,
    round(100*sum(days * content_creation_photo_edit_creation/total_duration)/sum(days),3) as content_creation_photo_edit_creation,
    round(100*sum(days * content_creation_video_audio_edit_creation/total_duration)/sum(days),3) as content_creation_video_audio_edit_creation,
    round(100*sum(days * content_creation_web_design_development/total_duration)/sum(days),3) as content_creation_web_design_development,
    round(100*sum(days * education/total_duration)/sum(days),3) as education,
    round(100*sum(days * entertainment_music_audio_streaming/total_duration)/sum(days),3) as entertainment_music_audio_streaming,
    round(100*sum(days * entertainment_other/total_duration)/sum(days),3) as entertainment_other,
    round(100*sum(days * entertainment_video_streaming/total_duration)/sum(days),3) as entertainment_video_streaming,
    round(100*sum(days * finance/total_duration)/sum(days),3) as finance,
    round(100*sum(days * games_other/total_duration)/sum(days),3) as games_other,
    round(100*sum(days * games_video_games/total_duration)/sum(days),3) as games_video_games,
    round(100*sum(days * mail/total_duration)/sum(days),3) as mail,
    round(100*sum(days * news/total_duration)/sum(days),3) as news,
    round(100*sum(days * unclassified/total_duration)/sum(days),3) as unclassified,
    round(100*sum(days * private/total_duration)/sum(days),3) as private,
    round(100*sum(days * productivity_crm/total_duration)/sum(days),3) as productivity_crm,
    round(100*sum(days * productivity_other/total_duration)/sum(days),3) as productivity_other,
    round(100*sum(days * productivity_presentations/total_duration)/sum(days),3) as productivity_presentations,
    round(100*sum(days * productivity_programming/total_duration)/sum(days),3) as productivity_programming,
    round(100*sum(days * productivity_project_management/total_duration)/sum(days),3) as productivity_project_management,
    round(100*sum(days * productivity_spreadsheets/total_duration)/sum(days),3) as productivity_spreadsheets,
    round(100*sum(days * productivity_word_processing/total_duration)/sum(days),3) as productivity_word_processing,
    round(100*sum(days * recreation_travel/total_duration)/sum(days),3) as recreation_travel,
    round(100*sum(days * reference/total_duration)/sum(days),3) as reference,
    round(100*sum(days * search/total_duration)/sum(days),3) as search,
    round(100*sum(days * shopping/total_duration)/sum(days),3) as shopping,
    round(100*sum(days * social_social_network/total_duration)/sum(days),3) as social_social_network,
    round(100*sum(days * social_communication/total_duration)/sum(days),3) as social_communication,
    round(100*sum(days * social_communication_live/total_duration)/sum(days),3) as social_communication_live
from (select a.guid,
             a.persona,
             count(b.*)                                        as days,
             sum(b.content_creation_photo_edit_creation)       as content_creation_photo_edit_creation,
             sum(b.content_creation_video_audio_edit_creation) as content_creation_video_audio_edit_creation,
             sum(b.content_creation_web_design_development)    as content_creation_web_design_development,
             sum(b.education)                                  as education,
             sum(b.entertainment_music_audio_streaming)        as entertainment_music_audio_streaming,
             sum(b.entertainment_other)                        as entertainment_other,
             sum(b.entertainment_video_streaming)              as entertainment_video_streaming,
             sum(b.finance)                                    as finance,
             sum(b.games_other)                                as games_other,
             sum(b.games_video_games)                          as games_video_games,
             sum(b.mail)                                       as mail,
             sum(b.news)                                       as news,
             sum(b.unclassified)                               as unclassified,
             sum(b.private)                                    as private,
             sum(b.productivity_crm)                           as productivity_crm,
             sum(b.productivity_other)                         as productivity_other,
             sum(b.productivity_presentations)                 as productivity_presentations,
             sum(b.productivity_programming)                   as productivity_programming,
             sum(b.productivity_project_management)            as productivity_project_management,
             sum(b.productivity_spreadsheets)                  as productivity_spreadsheets,
             sum(b.productivity_word_processing)               as productivity_word_processing,
             sum(b.recreation_travel)                          as recreation_travel,
             sum(b.reference)                                  as reference,
             sum(b.search)                                     as search,
             sum(b.shopping)                                   as shopping,
             sum(b.social_social_network)                      as social_social_network,
             sum(b.social_communication)                       as social_communication,
             sum(b.social_communication_live)                  as social_communication_live,
             sum(b.content_creation_video_audio_edit_creation +
                 b.content_creation_photo_edit_creation +
                 b.content_creation_web_design_development +
                 b.education +
                 b.entertainment_music_audio_streaming +
                 b.entertainment_other +
                 b.entertainment_video_streaming +
                 b.finance +
                 b.games_other +
                 b.games_video_games +
                 b.mail +
                 b.news +
                 b.unclassified +
                 b.private +
                 b.productivity_crm +
                 b.productivity_other +
                 b.productivity_presentations +
                 b.productivity_programming +
                 b.productivity_project_management +
                 b.productivity_spreadsheets +
                 b.productivity_word_processing +
                 b.recreation_travel +
                 b.reference +
                 b.search +
                 b.shopping +
                 b.social_social_network +
                 b.social_communication +
                 b.social_communication_live)                  as total_duration
      from reporting.system_sysinfo_unique_normalized a
               inner join
           reporting.system_web_cat_pivot_duration b
           on
               a.guid = b.guid
      group by 1, 2) inn
group by persona;







