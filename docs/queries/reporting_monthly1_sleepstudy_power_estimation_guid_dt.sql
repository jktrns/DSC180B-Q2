/* README
This script generates the reporting tables for the SleepStudy Power Estimation Report designed for all interested parties.
Objective is to analyze app-level power consumption through, SRUM power_estimation_data across recent_usage and scenario instances.

Target report: ICIP_SleepStudy_PowerEstimation
Date range & aggregation:
   Time range: 13 weeks
   Time aggregation: 13 weeks
   System aggregation: sysinfo_hash

Dependancies:
   ipip_windows_prod.mods_sleepstudy_power_estimation_data_13wks
   ipip_windows_prod.mods_sleepstudy_recent_usage_instance
   ipip_windows_prod.mods_sleepstudy_scenario_instance
   ipip_windows_analysis_pad.system_sysinfo_unique_normalized
   ipip_windows_analysis_pad.system_cpu_metadata

Output:
   ipip_windows_analysis_pad.reporting_mods_sleepstudy_system_sysinfo_unique_normalized
   ipip_windows_analysis_pad.reporting_mods_sleepstudy_unified_instances
   ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_apps
   ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_data_agg
   ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_users
*/

------------------------------------------------------------------------------------------------
-- Constrain date range and guid population for this report and create population hash groups --
------------------------------------------------------------------------------------------------
-- constrain time range
drop table if exists dt_range;
create temp table dt_range as
(
select
    cast( greatest(cast('03-02-2024' AS DATE), (now() - interval '91 days')) as DATE) as start_date
  , cast((now() - interval '1 day') AS DATE) as end_date);

-- identify guid sample common to both mods_sleepstudy_power_estimation_data and sysinfo. consolidate non top OEMs to reduce number of sysinfo hashes
drop table if exists selected_guids;
-- create temp table selected_guids as (
--     select a.guid,
--         sha256(
--             CAST(
--                 (b.chassistype || b.countryname_normalized || b.modelvendor_normalized || b.os || b.age_category || b.cpu_family || b.engagement_id || c.marketcodename) as bytea
--             )
--         ) as sysinfo_hash,
--         b.chassistype,
--         case
--             when b.countryname_normalized in (
--                 select countryname_normalized
--                 from reporting.system_sysinfo_unique_normalized
--                 group by countryname_normalized
--                 order by count(distinct guid) desc
--                 LIMIT 20
--             )
--             or b.countryname_normalized = 'Unknown' then b.countryname_normalized
--             else 'Other'
--         end as countryname_normalized,
--         case
--             when b.modelvendor_normalized in (
--                 select modelvendor_normalized
--                 from reporting.system_sysinfo_unique_normalized
--                 group by modelvendor_normalized
--                 order by count(distinct guid) desc
--                 LIMIT 20
--             )
--             or b.modelvendor_normalized = 'Unknown' then b.modelvendor_normalized
--             else 'Other'
--         end as modelvendor_normalized,
--         b.os,
--         b.age_category,
--         b.cpu_family,
--         b.engagement_id,
--         c.marketcodename
--     from (
--             select distinct guid
--             from university_analysis_pad.mods_sleepstudy_power_estimation_data_13wks
--         ) a
--         inner join reporting.system_sysinfo_unique_normalized b on a.guid = b.guid
--         left join reporting.system_cpu_metadata c on b.guid = c.guid
--     --order by random()
-- );

select count(distinct guid), count(*) from university_analysis_pad.mods_sleepstudy_power_estimation_data_13wks;

create temporary table selected guids
as
(

)


select count(*) from selected_guids;

--------------------------------------------------------------------------------------------------------
-- Grab prod data with guid and dt_range filters applied and perform any table-specific cleaning/prep --
--------------------------------------------------------------------------------------------------------
-- filter down prod table by guid/time, create acdc_event_proxy, clean-up type. <5 min
drop table if exists mods_sleepstudy_recent_usage_instance;
create temp table mods_sleepstudy_recent_usage_instance as (
    select distinct a.recent_usage_hash,
        a.guid,
        a.dt_utc,
        case
            when a.type ilike '%mask%' then 'Other'
            else a.type
        end as type,
        case
            when a.ac = 1 then 'AC_DISPLAY_ON'
            when a.ac = 0 then 'DC_DISPLAY_ON'
            else 'UN_DISPLAY_ON'
        end as acdc_event_name,
        a.duration as duration_s,
        a.screen_on_entry_reason,
        a.screen_on_exit_reason,
        case
            when a.energy_drain = 0 then null
            else CAST(a.energy_drain AS float)
        end as energy_drain,
        case
            when a.energy_drain < 0 then '<0'
            when a.energy_drain = 0 then '0'
            when a.energy_drain > 0 then '>0'
            else 'UNK'
        end as energy_drain_bin
    from ipip_windows_prod.mods_sleepstudy_recent_usage_instance a
        inner join selected_guids g
        on a.guid = g.guid
    where CAST(dt_utc as DATE) between (
            select start_date
            from dt_range ) and (
            select end_date
            from dt_range )
                and a.duration > 0
        );

-- filter down prod table by guid/time, pivot, create acdc_event_proxy, convert time units to sec. <5 min
drop table if exists mods_sleepstudy_scenario_instance;
create temp table mods_sleepstudy_scenario_instance as (
    select distinct sq.scenario_instance_hash,
        sq.guid,
        sq.dt_utc,
        sq.type,
        sq.ac,
        case
            when ac = 1 then 'AC_DISPLAY_OFF'
            when ac = 0 then 'DC_DISPLAY_OFF'
            else 'UN_DISPLAY_OFF'
        end as acdc_event_name,
        sq.duration_ms / 1e3 as duration_s,
        sq.cs_enter_reason,
        sq.cs_exit_reason,/home/layne/Documents/git/intel-queries/guids_on_off_suspend_time.sql
        case
            when sq.energy_drain = 0 then null
            else CAST(sq.energy_drain AS float)
        end as energy_drain,
        case
            when sq.energy_drain_raw like '-%' then '<0'
            when sq.energy_drain_raw = '0' then sq.energy_drain_raw
            when sq.energy_drain_raw ~ '^\\d+$' then '>0'
            else 'UNK'
        end as energy_drain_bin
    from (
            select a.scenario_instance_hash,
                a.guid,
                a.dt_utc,
                case
                    when a.type ilike '%mask%' then 'Other'
                    else a.type
                end as type,
                a.duration_ms,
                a.cs_enter_reason,
                a.cs_exit_reason,
                max(
                    case
                        when a.attribute_name = 'Ac' then CAST(a.attribute_value AS INT)
                    end
                ) as ac,
                max(
                    case
                        when a.attribute_name = 'EnergyDrain'
                        and a.attribute_value ~ '^-?\\d+$' then CAST(a.attribute_value AS INT)
                    end
                ) as energy_drain,
                max(
                    case
                        when a.attribute_name = 'EnergyDrain' then a.attribute_value
                    end
                ) as energy_drain_raw
            from ipip_windows_prod.mods_sleepstudy_scenario_instance_13wks a
                inner join selected_guids g on a.guid = g.guid
            where a.dt_utc between (
                    select start_date
                    from dt_range
                )
                and (
                    select end_date
                    from dt_range
                )
                and duration_ms > 0
            group by a.scenario_instance_hash,
                a.guid,
                a.dt_utc,
                a.type,
                a.duration_ms,
                a.cs_enter_reason,
                a.cs_exit_reason
        ) sq
);

-- remove 0 total_power_consumption records, extract app_id substring, and introduce dt_utc truncations. 21 min
drop table if exists mods_sleepstudy_power_estimation_data;
create temp table mods_sleepstudy_power_estimation_data as (
    select load_ts,
        batch_id,
        audit_zip,
        audit_internal_path,
        pe.guid,
        sg.sysinfo_hash,
        ts_utc,
        dt_utc,
        date_trunc('week', dt_utc) as wk_utc,
        date_trunc('month', dt_utc) as mn_utc,
        date_trunc('quarter', dt_utc) as qt_utc,
        ts_local,
        case
            when substring(app_id, 1, 1) = '\\' then regexp_substr(
                regexp_substr(app_id, '[^\\]+$'),
                '(.*)\\.(\\S*)'
            )
            else app_id
        end as app_id,
        case
            when user_id = 'UserIdMask' then 'USER'
            when user_id = 'Local Service' then upper(user_id)
            when user_id = 'SYUserIdMaskM' then 'SYSTEM'
            when user_id in (
                'LOCAL SERVICE',
                'NETWORK SERVICE',
                'SYSTEM',
                'NONE'
            ) then user_id
            else 'Other'
        end as user_id,
        cpu_power_consumption,
        display_power_consumption,
        disk_power_consumption,
        mbb_power_consumption,
        network_power_consumption,
        soc_power_consumption,
        loss_power_consumption,
        other_power_consumption,
        total_power_consumption,
        COALESCE(recent_usage_hash, scenario_instance_hash) as hash,
        case
            when scenario_instance_hash is not null then 'scenario instance'
            when recent_usage_hash is not null then 'recent usage instance'
            else 'parent instance not found'
        end as instance_type
    from ipip_windows_prod.mods_sleepstudy_power_estimation_data_13wks pe
        inner join selected_guids sg on pe.guid = sg.guid
    where (
            pe.dt_utc between (
                select start_date
                from dt_range
            )
            and (
                select end_date
                from dt_range
            )
        )
        and total_power_consumption > 0
        and (
            (scenario_instance_hash is not null)
            or (recent_usage_hash is not null)
        )
);

----------------------------------------------------------------
-- Join, profile, clean further, and perform desired analysis --
----------------------------------------------------------------
/* denormalize by joining in basic instance info from parents (scenario and recent_usage), standardize app_id, standardize. 30m */
drop table if exists mods_sleepstudy_power_estimation_data_denorm_int1;
create temp table mods_sleepstudy_power_estimation_data_denorm_int1 as (
    select distinct a.load_ts,
        a.batch_id,
        a.audit_zip,
        a.audit_internal_path,
        a.guid,
        a.ts_utc,
        a.dt_utc,
        a.wk_utc,
        a.mn_utc,
        a.qt_utc,
        a.app_id,
        a.user_id,
        a.cpu_power_consumption,
        a.display_power_consumption,
        a.disk_power_consumption,
        a.mbb_power_consumption,
        a.network_power_consumption,
        a.soc_power_consumption,
        a.loss_power_consumption,
        a.other_power_consumption,
        a.total_power_consumption,
        a.instance_type,
        a.hash,
        a.sysinfo_hash,
        COALESCE(s.type, r.type) as type,
        COALESCE(s.acdc_event_name, r.acdc_event_name) as acdc_event_name,
        COALESCE(s.duration_s, r.duration_s) as duration_s,
        COALESCE(s.cs_enter_reason, r.screen_on_entry_reason) as entry_reason,
        COALESCE(s.cs_exit_reason, r.screen_on_exit_reason) as exit_reason,
        COALESCE(s.energy_drain, r.energy_drain) as energy_drain,
        COALESCE(s.energy_drain_bin, r.energy_drain_bin) as energy_drain_bin
    from mods_sleepstudy_power_estimation_data a
        left join mods_sleepstudy_scenario_instance s on a.hash = s.scenario_instance_hash
        left join mods_sleepstudy_recent_usage_instance r on a.hash = r.recent_usage_hash
    where (s.scenario_instance_hash is not null)
        or (r.recent_usage_hash is not null)
);

/* profile and clean up high power_consumption instances */
-- profile total_power_consumption per app/user_id/acdc to determine outlier boundary
drop table if exists mods_sleepstudy_power_estimation_data_total_power_box_whisker;
create temp table mods_sleepstudy_power_estimation_data_total_power_box_whisker as (
    select sq.*,
        sq.q3 - sq.q1 as IQR
    from (
            select app_id,
                acdc_event_name,
                user_id,
                percentile_cont(0.25) within group (
                    order by total_power_consumption
                ) as q1,
                percentile_cont(0.75) within group (
                    order by total_power_consumption
                ) as q3
            from mods_sleepstudy_power_estimation_data_denorm_int1
            group by app_id,
                acdc_event_name,
                user_id
        ) sq
);

-- filter out instances with outlier total power or invalid EnergyDrain. introduce instance duration bin. 15 min
drop table if exists mods_sleepstudy_power_estimation_data_denorm_int2;
create temp table mods_sleepstudy_power_estimation_data_denorm_int2 as (
    select int1.load_ts,
        int1.batch_id,
        int1.audit_zip,
        int1.audit_internal_path,
        int1.guid,
        int1.ts_utc,
        int1.dt_utc,
        int1.wk_utc,
        int1.mn_utc,
        int1.qt_utc,
        int1.app_id,
        int1.user_id,
        int1.cpu_power_consumption,
        int1.display_power_consumption,
        int1.disk_power_consumption,
        int1.mbb_power_consumption,
        int1.network_power_consumption,
        int1.soc_power_consumption,
        int1.loss_power_consumption,
        int1.other_power_consumption,
        int1.total_power_consumption,
        int1.instance_type,
        int1.sysinfo_hash, --       ol.hash is null -- filter out high power outliers
        int1.type,
        int1.acdc_event_name,
        int1.duration_s,
        case
            when int1.duration_s / 60.0 < 10 then '0-10' --          when int1.duration_s / 60.0 < 60 then '10-60'
            when int1.duration_s / 60.0 >= 10 then '10+'
        end as duration_m_bin,
        int1.entry_reason,
        int1.exit_reason,
        int1.energy_drain,
        int1.energy_drain_bin,
        int1.hash,
        case
            when int1.total_power_consumption not between (pf.q1 - pf.IQR * 1.5) and (pf.q3 + pf.IQR * 1.5) then 1
            else 0
        end as outlier
    from mods_sleepstudy_power_estimation_data_denorm_int1 int1 --         left join #mods_sleepstudy_power_estimation_outlier_instances ol
        left join mods_sleepstudy_power_estimation_data_total_power_box_whisker pf on int1.acdc_event_name = pf.acdc_event_name
        and int1.user_id = pf.user_id
        and int1.app_id = pf.app_id
    where int1.energy_drain_bin in ('0', '>0') -- exclude instances with negative or null EnergyDrains for data volume management
        and int1.acdc_event_name !~ '^UN' -- exclude instances with unknown AC/DC state for data volume management
);

/* profile and clean up apps by creating Other group */
-- apps aggregated
drop table if exists mods_sleepstudy_power_estimation_app;
create temp table mods_sleepstudy_power_estimation_app as (
    select app_id,
        count(distinct guid) as guid_count,
        sum(total_power_consumption * duration_s) / sum(duration_s) as total_power_consumption_wavg,
        sum(
            total_power_consumption / energy_drain * duration_s
        ) / sum(duration_s) as total_power_consumption_norm_wavg -- computed only for instances with positive energy drain. null when energy_drain = 0
    from mods_sleepstudy_power_estimation_data_denorm_int2
    group by app_id
);

-- guid_counts by app. 4.8m app_id
drop table if exists mods_sleepstudy_power_estimation_app_rnk;
create temp table mods_sleepstudy_power_estimation_app_rnk as (
    select *,
        row_number() over (
            order by guid_count desc
        ) as rnk_guid_count,
        row_number() over (
            order by total_power_consumption_wavg desc
        ) as rnk_total_power_consumption_wavg,
        case
            when total_power_consumption_norm_wavg is not null then row_number() over (
                order by total_power_consumption_norm_wavg desc
            )
        end as rnk_total_power_consumption_norm_wavg,
        row_number() over (
            partition by lower(app_id)
            order by guid_count desc
        ) as rnk_lower_app_id -- deal with case sensitivity. take only top app_id per lower(app_id). Put the rest in Other
    from mods_sleepstudy_power_estimation_app
);

-- identify top apps by various measures
drop table if exists mods_sleepstudy_power_estimation_app_top;
create temp table mods_sleepstudy_power_estimation_app_top as (
    (
        select app_id
        from mods_sleepstudy_power_estimation_app
        where guid_count > 10000
        order by guid_count desc
        LIMIT 300
    )
    union
    (
        select app_id
        from mods_sleepstudy_power_estimation_app
        where guid_count > 10000
        order by total_power_consumption_wavg desc
        LIMIT 300
    )
    union
    (
        select app_id
        from mods_sleepstudy_power_estimation_app
        where guid_count > 10000
        order by total_power_consumption_norm_wavg desc
        LIMIT 300
    )
);

-- create Other app for long tail of sparse apps. Introduce app_type
drop table if exists mods_sleepstudy_power_estimation_app_map;
create temp table mods_sleepstudy_power_estimation_app_map as (
    select sq.*,
        sha256(CAST(app_id_other as bytea)) as app_id_other_hash, -- hash to deal with PowerBI case-insensitivity for primary keys
        case
            when sq.app_id_other = 'Other' then 'Multiple'
            when c.app_type is not null then c.app_type
            else 'UNK'
        end as app_type
    from (
            select r.app_id,
                r.guid_count,
                case
                    when r.app_id in (
                        'Unknown',
                        'UserIdMask',
                        'esrv.exe',
                        'esrv_svc.exe',
                        'SurSvc.exe',
                        'IntelSoftwareAssetManagerService.exe'
                    ) then r.app_id
                    when t.app_id is not null then r.app_id
                    else 'Other'
                end as app_id_other
            from mods_sleepstudy_power_estimation_app_rnk r
                left join mods_sleepstudy_power_estimation_app_top t on r.app_id = t.app_id
        ) sq
        left join metadata.apps_execlass_combined c on lower(sq.app_id) = c.proc_name
);

-- map app_id_subtr to app_id_other. introduce aggregate instance hash to reduce row count later. 15m
drop table if exists mods_sleepstudy_power_estimation_data_denorm_int3;
create temp table mods_sleepstudy_power_estimation_data_denorm_int3 as (
    select pe.load_ts,
        pe.batch_id,
        pe.audit_zip,
        pe.audit_internal_path,
        pe.guid,
        pe.sysinfo_hash, -- replaces guid
        pe.instance_type, -- recent_usage or scenario
        pe.ts_utc,
        pe.dt_utc,
        pe.wk_utc,
        pe.mn_utc,
        pe.qt_utc,
        pe.app_id, -- replace with consolidated app groups
        map.app_id_other, -- same as app_id but uncommon apps grouped into Other
        map.app_id_other_hash,
        map.app_type,
        pe.user_id,
        pe.type, -- active, screen off, sleep
        pe.acdc_event_name,
        pe.duration_s,
        pe.duration_m_bin,
        pe.energy_drain,
        pe.energy_drain_bin,
        pe.hash,
        pe.cpu_power_consumption,
        pe.display_power_consumption,
        pe.disk_power_consumption,
        pe.mbb_power_consumption,
        pe.network_power_consumption,
        pe.soc_power_consumption,
        pe.loss_power_consumption,
        pe.other_power_consumption,
        pe.total_power_consumption,
        pe.outlier,
        sha256(
            CAST(
                sysinfo_hash || instance_type || type || acdc_event_name || energy_drain_bin as bytea
            )
        ) as si_itype_type_acdc_drain_hash
    from mods_sleepstudy_power_estimation_data_denorm_int2 pe
        left join mods_sleepstudy_power_estimation_app_map map on pe.app_id = map.app_id
    where (
            (
                pe.acdc_event_name ~ '^AC'
                and pe.energy_drain_bin = '0'
            ) -- battery must not be draining while on AC. 10% of AC instances violate this
            or (
                pe.acdc_event_name ~ '^DC'
                and pe.energy_drain_bin = '>0'
            )
        ) -- battery must be draining while on DC. 2.5% of DC instances violate this
        and pe.outlier = 0
);

-----------------------------
-- Update reporting tables --
-----------------------------
-- create sysinfo hash table using aggregated groups and compute guid_count
drop table if exists ipip_windows_analysis_pad.reporting_mods_sleepstudy_system_sysinfo_unique_normalized;
create table ipip_windows_analysis_pad.reporting_mods_sleepstudy_system_sysinfo_unique_normalized as
(
select
    now() as load_ts
  , sysinfo_hash
  , chassistype
  , countryname_normalized
  , modelvendor_normalized
  , os
  , age_category
  , cpu_family
  , engagement_id
  , marketcodename
  , count(distinct guid) as guid_count
from
    selected_guids
group by
    sysinfo_hash, chassistype, countryname_normalized, modelvendor_normalized, os, age_category, cpu_family
                , engagement_id, marketcodename );

-- create app table aggregating on consolidated app_id. 5 min
drop table if exists ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_apps;
create table ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_apps as (
    select app_id_other_hash,
        app_id_other,
        app_type,
        count(distinct int3.guid) as guid_count,
        count(distinct int3.hash) as hash_count,
        sum(int3.duration_s / 3600.0) as duration_h
    from mods_sleepstudy_power_estimation_data_denorm_int3 int3
    group by app_id_other_hash,
        app_id_other,
        app_type
);

-- aggregate on aggregate instance hash. 1 min
drop table if exists ipip_windows_analysis_pad.reporting_mods_sleepstudy_unified_instances;
create table ipip_windows_analysis_pad.reporting_mods_sleepstudy_unified_instances as (
    select distinct si_itype_type_acdc_drain_hash,
        sysinfo_hash,
        instance_type,
        type,
        acdc_event_name,
        energy_drain_bin,
        sum(duration_s) as duration_s,
        sum(duration_s) / 3600.0 as duration_h,
        count(distinct hash) as ninstance
    from mods_sleepstudy_power_estimation_data_denorm_int3
    group by si_itype_type_acdc_drain_hash,
        sysinfo_hash,
        instance_type,
        type,
        acdc_event_name,
        energy_drain_bin
);

-- aggregate on user_id. 2 min
drop table if exists ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_users;
create table ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_users as (
    select user_id,
        count(distinct guid) as guid_count,
        count(distinct hash) as hash_count,
        sum(duration_s) / 3600.0 as duration_h
    from mods_sleepstudy_power_estimation_data_denorm_int3
    group by user_id
);

-- aggregate on aggregate instance hash and app_id_other. 1 min
drop table if exists ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_data_agg;
create table ipip_windows_analysis_pad.reporting_mods_sleepstudy_power_estimation_data_agg as (
    select si_itype_type_acdc_drain_hash,
        user_id,
        app_id_other_hash,
        outlier,
        count(distinct hash) as hash_count,
        /*for weighted average powers*/
        sum(cpu_power_consumption * duration_s) as cpu_power_consumption_x_duration_s,
        sum(display_power_consumption * duration_s) as display_power_consumption_x_duration_s,
        sum(disk_power_consumption * duration_s) as disk_power_consumption_x_duration_s,
        sum(mbb_power_consumption * duration_s) as mbb_power_consumption_x_duration_s,
        sum(network_power_consumption * duration_s) as network_power_consumption_x_duration_s,
        sum(soc_power_consumption * duration_s) as soc_power_consumption_x_duration_s,
        sum(loss_power_consumption * duration_s) as loss_power_consumption_x_duration_s,
        sum(other_power_consumption * duration_s) as other_power_consumption_x_duration_s,
        sum(total_power_consumption * duration_s) as total_power_consumption_x_duration_s,
        sum(duration_s) as duration_s,
        /*for simple average powers (unweighted)*/
        sum(cpu_power_consumption) as cpu_power_consumption_sum,
        sum(display_power_consumption) as display_power_consumption_sum,
        sum(disk_power_consumption) as disk_power_consumption_sum,
        sum(mbb_power_consumption) as mbb_power_consumption_sum,
        sum(network_power_consumption) as network_power_consumption_sum,
        sum(soc_power_consumption) as soc_power_consumption_sum,
        sum(loss_power_consumption) as loss_power_consumption_sum,
        sum(other_power_consumption) as other_power_consumption_sum,
        sum(total_power_consumption) as total_power_consumption_sum,
        count(*) as record_count,
        /*for weighted average normalized powers*/
        sum(
            cpu_power_consumption / energy_drain * duration_s
        ) as cpu_power_consumption_norm_x_duration_s,
        sum(
            display_power_consumption / energy_drain * duration_s
        ) as display_power_consumption_norm_x_duration_s,
        sum(
            disk_power_consumption / energy_drain * duration_s
        ) as disk_power_consumption_norm_x_duration_s,
        sum(
            mbb_power_consumption / energy_drain * duration_s
        ) as mbb_power_consumption_norm_x_duration_s,
        sum(
            network_power_consumption / energy_drain * duration_s
        ) as network_power_consumption_norm_x_duration_s,
        sum(
            soc_power_consumption / energy_drain * duration_s
        ) as soc_power_consumption_norm_x_duration_s,
        sum(
            loss_power_consumption / energy_drain * duration_s
        ) as loss_power_consumption_norm_x_duration_s,
        sum(
            other_power_consumption / energy_drain * duration_s
        ) as other_power_consumption_norm_x_duration_s,
        sum(
            total_power_consumption / energy_drain * duration_s
        ) as total_power_consumption_norm_x_duration_s
    from mods_sleepstudy_power_estimation_data_denorm_int3
    group by si_itype_type_acdc_drain_hash,
        user_id,
        app_id_other_hash,
        outlier
);
-- 77m rows

-------------------
-- End of script --
-------------------

------------------------------
-- Development Scratch Area --
------------------------------
-- profile power_consumptions. very heavy query. only run once in a while to select and hard-code an outlier boundary. 2h+
-- drop table if exists #mods_sleepstudy_power_estimation_data_consumption_profile;
-- create temp table #mods_sleepstudy_power_estimation_data_consumption_profile as
-- (
-- select
--     consumption_type
--   , sum(consumption_mW * duration_s) / sum(duration_s)               as consumption_mW_wavg
--   , min(consumption_mW)                                              as min_consumption_mW
--   , percentile_cont(0.01) within group ( order by consumption_mW )   as pct01
--   , percentile_cont(0.05) within group ( order by consumption_mW )   as pct05
--   , percentile_cont(0.10) within group ( order by consumption_mW )   as pct10
--   , percentile_cont(0.25) within group ( order by consumption_mW )   as pct25
--   , percentile_cont(0.50) within group ( order by consumption_mW )   as pct50
--   , percentile_cont(0.75) within group ( order by consumption_mW )   as pct75
--   , percentile_cont(0.90) within group ( order by consumption_mW )   as pct90
--   , percentile_cont(0.95) within group ( order by consumption_mW )   as pct95
--   , percentile_cont(0.99) within group ( order by consumption_mW )   as pct99
--   , percentile_cont(0.999) within group ( order by consumption_mW )  as pct999
--   , percentile_cont(0.9999) within group ( order by consumption_mW ) as pct9999 -- use this to identify outlier instances and apps
--   , max(consumption_mW)                                              as max_consumption_mW
-- from
--     #mods_sleepstudy_power_estimation_data_denorm_int1 unpivot(consumption_mW for consumption_type in (
--     cpu_power_consumption
--   , display_power_consumption
--   , disk_power_consumption
--   , mbb_power_consumption
--   , network_power_consumption
--   , soc_power_consumption
--   , loss_power_consumption
--   , other_power_consumption
--   , total_power_consumption))
-- group by
--     consumption_type );

-- identify outlier hashes based on total_power_consumption percentile
-- drop table if exists #mods_sleepstudy_power_estimation_outlier_instances;
-- create temp table #mods_sleepstudy_power_estimation_outlier_instances distkey (guid)
--                                                                       sortkey (guid, dt_utc, hash) as
-- (
-- select distinct
--     hash
--   , guid
--   , dt_utc
-- from
--     #mods_sleepstudy_power_estimation_data_denorm_int1 i1
-- where
--     total_power_consumption > 10000 -- determined from power_consumption profiling analysis as the 99.9th percentile
--         total_power_consumption > (
--                                   select
--                                       pct999
--                                   from
--                                       #mods_sleepstudy_power_estimation_data_consumption_profile
--                                   where
--                                       consumption_type = 'total_power_consumption' )
-- );

-- compute app guid_count percentiles for all apps
-- drop table if exists #mods_sleepstudy_power_estimation_app_profile;
-- create temp table #mods_sleepstudy_power_estimation_app_profile diststyle all as
-- (
-- select
--     percentile_cont(0.001) within group ( order by guid_count )  as pct001 -- 1. least common apps
--   , percentile_cont(0.01) within group ( order by guid_count )   as pct01
--   , percentile_cont(0.05) within group ( order by guid_count )   as pct05
--   , percentile_cont(0.10) within group ( order by guid_count )   as pct10
--   , percentile_cont(0.25) within group ( order by guid_count )   as pct25
--   , percentile_cont(0.50) within group ( order by guid_count )   as pct50
--   , percentile_cont(0.75) within group ( order by guid_count )   as pct75
--   , percentile_cont(0.90) within group ( order by guid_count )   as pct90
--   , percentile_cont(0.95) within group ( order by guid_count )   as pct95
--   , percentile_cont(0.99) within group ( order by guid_count )   as pct99
--   , percentile_cont(0.999) within group ( order by guid_count )  as pct999 -- 8.8k
--   , percentile_cont(0.9999) within group ( order by guid_count ) as pct9999 -- 306k. most common apps reside in the very top percentile > 0.99
-- from
--     #mods_sleepstudy_power_estimation_app_rnk app );