# Reporting Schema Table Definition

[TOC]

## Introduction

This schema was introduced in an effort to make the DCA Telemetry data more digestible/query-able for a LLM. Towards that aim, several changes were made:

* Virtually all tables now are JOINable via GUID
* Most of the tables have a **dt** (date) column
* The tables are almost ALL indexed by **guid** and, if available, **dt**, on the assumption that most of the joins will be based on those columns
* Generally, the *key-value* format of some of the telemetry data has been avoided. Only one table, the **system_network_consumption** table still has a key-value(s) format. This is due to the number of metrics collected for that particular key.

## Tables

### system_sysinfo_unique_normalized

This table is unmodified from the raw DCA data. There was no need to modify it. 

In short, this is a client metadata table, providing important attributes about the client system, identified by the guid column. In here, you can find the country of origin for the data, the model vendor (OEM), the amount of RAM memory, the processor, the video card, the persona classification of the client, etc. This is a table that you should join to any of the other fact tables in this **reporting** schema to provide meaningful results.

| Column                    | Type      | Description                                                  |
| ------------------------- | --------- | ------------------------------------------------------------ |
| load_ts                   | timestamp | Timestamp the data was loaded into the table. Not as applicable for this use case. |
| guid                      | text      | Globally Unique IDentifier assigned to each and every client system in the DCA dataset. |
| chassistype               | text      | One of 7 identifiers for the form factor of the client: **2 in 1, Desktop, Intel NUC/STK, Notebook, Other, Server/WS, Tablet** |
| chassistype_2in1_category | text      | In the sub-category of 2-in-1 systems, there are 3 classifications provided: **Convertible, Detachable, Unknown** |
| countryname               | text      | Country the data originated from                             |
| countryname_normalized    | text      | Cleansed country the data originated from                    |
| modelvendor               | text      | The manufacturer of the client (OEM)                         |
| modelvendor_normalized    | text      | Cleansed modelvendor column. Manufactured of the client (OEM) |
| model                     | text      | The model identifier for the client                          |
| model_normalized          | text      | Cleansed model identifier for the client                     |
| ram                       | real      | The amount of client RAM, in GB                              |
| os                        | text      | Operating system, like Win10, Win11, etc.                    |
| #ofcores                  | text      | The number of cores in the processor of the client           |
| age_category              | text      | Binned category for the age of the client. Note: This is unreliable now, since the data is very old at this point |
| graphicsmanuf             | text      | Manufacturer (OEM) of the graphics solution of the client (Intel, Nvidia, etc.) |
| gfxcard                   | text      | The primary graphics card in the client (Intel(R) UHD Graphics 630, NVIDIA GeForce RTX 2060, etc.) |
| graphicscardclass         | text      | Binning of the graphics card by performance (Low-Midrange Graphics Cards, Office-Class Graphics Cards, etc.) Note: this data is stale and should NOT be used |
| processornumber           | text      | The lithography of the processor (14 nm, 10 nm, etc.)        |
| cpuvendor                 | text      | Manufacturer of the processor (Intel, AMD, etc.)             |
| cpuname                   | text      | The generation of class of processor (9th Gen i9, Xeon, etc.) |
| cpucode                   | text      | The marketing model identifier of the processor (i5-6300U, i9-9900, E3-1230v2, etc.) |
| cpu_family                | text      | Identifier for the family of the processor (Core i5, Core i9, etc.) |
| cpu_suffix                | text      | Identifier for a sub-identifier of the processor (Core-U, etc.), usually designating a sub-class of the processor (mobile, etc.) |
| screensize_category       | text      | Rough estimate of the screen width of the client (15x, >24x, etc.) |
| persona                   | text      | Derived column based on foreground application usage. There are 11 different persona classifications: <br>Casual Gamer<br/>Casual User<br/>Communication<br/>Content Creator/IT<br/>Entertainment<br/>File & Network Sharer<br/>Gamer<br/>Office/Productivity<br/>Unknown<br/>Web User<br/>Win Store App User |
| processor_line            | text      | Similar to cpu_suffix, a sub-designation of the type of processor (U-processor, M-processor, etc.) |
| vpro-enabled              | text      | Whether or not the client has Intel V-Pro technology enabled (Y, N) |
| firstreportdate           | timestamp | First sighting of the client in the DCA dataset              |
| lastreportdate            | timestamp | Last known sighting of the client in the DCA dataset         |
| discretegraphics          | text      | Does the client have a discrete graphics solution, or an embedded solution (Y, N) |
| cpu_stepping              | text      | Intel designation for the processor in the form of Family, Model, and Stepping values: Intel64 Family 6 Model 142 Stepping 10, Intel64 Family 6 Model 37 Stepping 5, etc. |
| engagement_id             | text      | Intel DCA data comes from various deployments of the DCA collector. This identifies the deployment: Consumer - IDSA, Consumer - GAMEPLAY, etc. |

### system_batt_dc_events

Summarized battery utilization statistics for clients while running on battery. Summarized by **guid** and **dt**.

| Column                        | Type      | Description                                                  |
| ----------------------------- | --------- | ------------------------------------------------------------ |
| load_ts                       | timestamp | Time stamp for when the data was loaded into the table       |
| guid                          | text      | Globally Unique IDentifier assigned to each and every client system in the DCA dataset. |
| dt                            | date      | Date the data was collected (local time)                     |
| duration_mins                 | integer   | Duration for the DC event in minutes                         |
| max_power_on_battery_percent  | integer   | Maximum battery percentage for power on event                |
| min_power_on_battery_percent  | integer   | Minimum battery percentage for power on event                |
| avg_power_on_battery_percent  | integer   | Average (mean) battery percentage for power on event         |
| max_power_off_battery_percent | integer   | Maximum battery percentage for power off event               |
| min_power_off_battery_percent | integer   | Minimum battery percentage for power off event               |
| avg_power_off_battery_percent | integer   | Average (mean) battery percentage for power off event        |
| num_power_ons                 | integer   | The number of power ons on this date                         |

### system_cpu_metadata

Specifications about the processor for the client. This is purely a **guid** based dataset.

| Column              | Type | Description                                                  |
| ------------------- | ---- | ------------------------------------------------------------ |
| guid                | text | Globally Unique IDentifier assigned to each and every client system in the DCA dataset. |
| cpu                 | text | Identifier string for Processor such as one would see in Windows System Information (Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz, etc.) |
| cpucode             | text | Shortened form of the cpu identifier (i5-8250U, etc.)        |
| processortechnology | text | Internal manufacturing process used to make the processor (1272, etc.) The 12 indicates it was built on a 12 inch wafer. |
| lithography         | text | The lithography utilized to make the processor (14nm, 10nm, etc.) which is an description of the width of the circuit "lines" in the processor. Smaller lithographies indicate newer processes. |
| marketsegment       | text | Bin identifying the segment the processor was marketed for:<br />MBL - mobile<br />DT - desktop<br />EMB - embedded<br />SRV - server<br />WS - workstation<br />Unknown - unknown, usually assigned to a processor not made by Intel. |
| cpugen              | text | The "generation" of the processor (11th Gen i7, Pentium/Celeron-1st Gen, etc.) A general binning column |
| launchdate          | text | Date the processor was released to the public                |
| estfirstusedt       | text | Estimated date the processor was first used                  |
| marketcodename      | text | Internal Intel code name for the processor (Rocket Lake, Arrandale, etc.) |
| #ofcores            | text | Number of processing cores in the processor (does NOT account for Intel Hyperthreading.) |
| spec_tdp            | text | The Thermal Design Power of the processor, in Watts. Thermal Design Power (TDP), also known as thermal design point, is the maximum amount of heat can generate and that its cooling system is designed to dissipate during normal operation at a non-turbo clock rate (base frequency). |

### system_display_devices

This table describes the usage of the display devices utilized by the client system. It provides information like the number of devices connected to the client (which can be determined by the number of distinct **display_id**s that the system is utilizing), the port a display is using, the resolution in pixels, the refresh rate of the display, and the duration of time on AC (client plugged in) and on DC (client running of internal battery).

Note that this table has a finer temporal granularity than simply **guid, dt**. It provides timestamps **ts** for each event the client experienced with the display. Simple summarization is possible by simply aggregating by **guid, dt**.

| Column            | Type             | Description                                                  |
| ----------------- | ---------------- | ------------------------------------------------------------ |
| load_ts           | timestamp        | Time stamp for when the data was loaded into the table       |
| guid              | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset. |
| dt                | date             | Date the data was collected (local time)                     |
| ts                | timestamp        | Timestamp for when this event was collected                  |
| display_id        | bigint           | Id assigned to this display device by ATL. The only real value this column has for analysis is in enumerating the displays connected to the client system. |
| adapter_id        | bigint           | OS assigned (64 bit) identifier for adapter to which this display is connected. One adapter can have multiple ports. Adapter ids assigned by OS can change across reboots for same physical adapter. |
| port              | text             | Port on which this display is connected. Port can have one of below values: "PORT_A", "PORT_B", "PORT_C", "PORT_D", "PORT_E", "PORT_F", "PORT_G", "PORT_H", "PORT_I", "DSI_PORT_0", "DSI_PORT_1", "COLLAGE_PORT_0", "UNKNOWN". One port can support multiple connectors of same type |
| sink_index        | bigint           | 0 based index assigned by display driver to distinguish between multiple connectors of same type on same port |
| connection_type   | text             | Interface used to connect this display device. This field should not be empty/null. This can have one of the below values as returned by OS:"OTHER", "HD15", "SVIDEO", "COMPOSITE_VIDEO", "COMPONENT_VIDEO", "DVI", "HDMI", "LVDS", "D_JPN", "SDI", "DP_EXTERNAL", "DP_EMBEDDED", "UDI_EXTERNAL", "UDI_EMBEDDED", "SDTVDONGLE", "MIRACAST", "INDIRECT_WIRED", "INDIRECT_VIRTUAL", "INTERNAL" |
| vendor_name       | text             | Vendor name of display device. This can have one of the below values: "Dell", "BenQ", "Asus", "Acer", "HP", "Lenovo", "LG", "Sony", "Samsung","Philips", "Viewsonic", "AOC", "zebronics", "MSI", "VIZIO", "TCL", "Sansui", "Kodak", "Oneplus", "Hisense","Supersonic", "Redmi","MI","Other", "Unknown".("Unknown" indicates name collected from system was null/empty. "Other" indicates name was non null but it did not match any name from list of known vendors) |
| status            | bigint           | Status of connected display: 1: Display connected & active; 0: Display connected but inactive |
| resolution_width  | bigint           | Resolution width set on display. When "Status" is inactive, this value will be 0 |
| resolution_heigth | bigint           | Resolution height set on display. When "Status is inactive, this value will be 0 (note, height is misspelled in this column name.) |
| refresh_rate      | double precision | Refresh rate set on display. When "Status" is inactive, this value will be 0 |
| duration_ac       | bigint           | Duration in seconds for which this combination of settings was set on the display and system was on ac power |
| duration_dc       | bigint           | Duration in seconds for which this combination of settings was set on the display and system was on dc power |

### system_frgnd_apps_types

This table provides a summarization of the foreground applications detected on a particular client on a particular date (**dt**). It provides the applications detected (**exe_name**), the classification for the application (**app_type**), and some relevant telemetry data about the application.

| Column            | Type             | Description                                                  |
| ----------------- | ---------------- | ------------------------------------------------------------ |
| guid              | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset. |
| dt                | date             | Date the data was collected (local time)                     |
| app_type          | text             | A binned category for the type of application this is. The categories are: Communication<br/>Developer tools<br/>Entertainment<br/>Gaming<br/>Gaming auxiliary<br/>Installers + Updaters<br/>Lifestyle<br/>Multimedia<br/>Other<br/>Productivity<br/>Security<br/>Social<br/>System<br/>Utilities + tools<br/>Web Browsing. <br />There is also the possibility of NULL values in the case where the application was unknown. |
| exe_name          | text             | The name of the process that was detected by the DCA collector. |
| company_short     | text             | The name of the company (if known) that produced the process (**exe_name**). |
| totalsecfocal_day | double precision | The total amount focal screen time the process consumed during the date in seconds. |
| avg_fract_desktop | double precision | The proportion of the screen (between 0 and 1) that the process took on the screen |
| process_desc      | text             | A brief description of the process (if known).               |
| lines_per_day     | bigint           | The number of times the process was in focus on the screen during the day on the date in question. |

### system_frgnd_daily_summary

Another summary table for foreground application data, similar in content to the **system_frgnd_apps_types** table. It lacks the application types and descriptions the **system_frgnd_apps_types** provides.

| Column                 | Type             | Description                                                  |
| ---------------------- | ---------------- | ------------------------------------------------------------ |
| load_ts                | timestamp        | Time stamp for when the data was loaded into the table       |
| guid                   | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset. |
| dt                     | date             | Date the data was collected (local time)                     |
| proc_name              | text             | The name of the process that was detected by the DCA collector. Identical to the **exe_name** in the **system_frgnd_apps_types** table. |
| proc_package           | text             | Package name for the executable if available                 |
| captioned              | text             | Caption state of the application. Typically, c0 applications are full screen, and thus metro apps in Win 8.x+ |
| window_mode            | text             | Calculated by ETL - sourced from captioned - Mode of a display device, Window mode or Full mode. E.g. W or F |
| total_dur_sec          | double precision | Duration of the time the process/application is in foreground |
| wght_avg_fract_desktop | double precision | Percentage of the screen used to show the program. Negative 1 (-1) if WindowPixelSize is non-zero and no value for DesktopPixelSize. Zero (0) if WindowPixelSize is null or zero |
| num_pid                | bigint           | The sum total of the number of distinct PIDs (Process IDentifier) that was counted by the collector |
| lines_per_day          | bigint           | The number of times the process was in focus on the screen during the day on the date in question. |

### system_hw_pkg_power

This table provides a statistical summary of the amount of power a processor consumes on a daily basis. It provides the number of samples taken for the summary, as well as the mean (average) and maximum power it saw during the date (**dt**) in question.

| Column   | Type             | Description                                                  |
| -------- | ---------------- | ------------------------------------------------------------ |
| load_ts  | timestamp        | Time stamp for when the data was loaded into the table       |
| guid     | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt       | date             | Date the data was collected (local time)                     |
| instance | integer          | The enumeration of the processor in the client. Typically, there is only one processor in a client system. Workstations and servers CAN enumerate more than one processor |
| nrs      | integer          | The number of samples collected for the statistical summary of the data. Note: the DCA collector samples every five (5) seconds, so to estimate the amount of time for the statistical summary, multiple the **nrs** value by 5. |
| mean     | double precision | The mean (average) power consumed by this particular client processor, in Watts |
| max      | double precision | The maximum power consumed by this particular client processor, in Watts |

### system_memory_utilization

This table provides a statistical summary of the RAM memory utilization of the client system on a daily basis, for the date (**dt**) in question. It will provide average free memory available, in megabytes (MB, the RAM capacity in MB, the average utilized RAM in MB, and the percentage of memory used.

| Column              | Type             | Description                                                  |
| ------------------- | ---------------- | ------------------------------------------------------------ |
| load_ts             | timestamp        | Time stamp for when the data was loaded into the table       |
| guid                | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                  | date             | Date the data was collected (local time)                     |
| nrs                 | integer          | The number of samples collected for the statistical summary of the data. Note: the DCA collector samples every five (5) seconds, so to estimate the amount of time for the statistical summary, multiple the **nrs** value by 5. |
| avg_free_ram        | double precision | The average amount of free RAM memory on the date (**dt**), in megabytes (MB) |
| sysinfo_ram         | integer          | The total amount of RAM available on the client, in MB       |
| utilized_ram        | double precision | The average amount of utilized RAM memory, in MB, for the date |
| avg_percentage_used | double precision | The average percentage of the RAM on the system used during the day |

### system_mods_power_consumption

This is a table of data generated from the modern sleepstudy data report of the Microsoft Windows **powercfg** command. It provides *estimates* for the amount of power consumed by all the processes running on the client. The power consumption estimates are in mW (milliwatts), and are categorized into different dimensions of power consumption for a client system.

| Column                    | Type      | Description                                                  |
| ------------------------- | --------- | ------------------------------------------------------------ |
| load_ts                   | timestamp | Time stamp for when the data was loaded into the table       |
| guid                      | text      | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                        | date      | Date the data was collected (local time)                     |
| app_id                    | text      | Process/application name. The processes in this table are more comprehensive than those provided in the frgnd tables; this is an enumeration of ALL the processes the Windows client observed |
| user_id                   | text      | This is an enumeration of what type of process or account the application was running under. The enumeration is the following:<br />NETWORK SERVICE - system level network process<br/>NONE - unknown<br/>LOCAL SERVICE - local service for the process<br/>UserIdMask - ran specifically as the user<br/>SYSTEM - ran as a Os/system level process |
| cpu_power_consumption     | integer   | power in mW the cpu/processor consumed for this process/application |
| display_power_consumption | integer   | power in mW the display consumed for this process/application |
| disk_power_consumption    | integer   | power in mW the disk/storage consumed for this process/application |
| mbb_power_consumption     | integer   | power in mW the main board consumed for this process/application |
| network_power_consumption | integer   | power in mW the networking consumed for this process/application |
| soc_power_consumption     | integer   | power in mW the "system on chip" consumed for this process/application |
| loss_power_consumption    | integer   | power in mW lost for this process/application                |
| other_power_consumption   | integer   | power in mW uncategorized consumed for this process/application |
| total_power_consumption   | integer   | total power in mW this process consumed across ALL categories for this process/application |

### system_mods_top_blocker_hist

This data table provides a event-based listing of all "blockers" the system encountered. It is actually a fairly obscure dataset, and not one readily analyzed. Being event-based, this dataset has a time stamp (**ts_local**) it uses to mark the moment the blocker occurred.

| Column         | Type             | Description                                                  |
| -------------- | ---------------- | ------------------------------------------------------------ |
| load_ts        | timestamp        | Time stamp for when the data was loaded into the table       |
| guid           | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| ts_utc         | timestamp        | Time the event occurred in UTC time                          |
| dt_utc         | date             | Date the event occurred in UTC time                          |
| ts_local       | timestamp        | Time the event occurred in local time for the client         |
| dt             | date             | Date the event occurred in local time for the client         |
| blocker_name   | text             | Text description of the blocker. It will either be a system level type of label, or a specific description of a process or hardware component that is the source of the block |
| active_time_ms | double precision | Amount of time for the block                                 |
| activity_level | text             | Description of the activity of the system (high, moderate, or low) |
| blocker_type   | text             | The type of blocker. The allowed values are: <br />Activator<br/>Audio Activity<br/>Fx Device<br/>PDC Phase<br/>PEP Pre-Veto<br/>Processor |
| blocker_id     | text             | Just the blocker id. Limited to no analytic value            |

### system_network_consumption

A daily level summarization of the network consumption for the system, in a key-value(s) format. For each guid and date (**dt**), there will *usually* be two rows: one for OS:NETWORK INTERFACE::BYTES SENT/SEC::, and the other for OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::. It does NOT discriminate between the different types of network connection (WIFI, etc.) It is purely a summarization of ALL bytes sent and received in terms of bytes/second.

To estimate the total number of bytes sent or received, one has to utilize the **nrs** column, which was the number of samples taken on that day. Note again, the DCA collector samples every 5 seconds, so to get an estimate of the amount of time for this date, you have to multiple the **nrs** column by 5. Subsequently, after that, you can multiple that result by the **avg_bytes_sec** to get an estimate of the total bytes/second.

| Column        | Type             | Description                                                  |
| ------------- | ---------------- | ------------------------------------------------------------ |
| load_ts       | timestamp        | Time stamp for when the data was loaded into the table       |
| guid          | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt            | date             | Date the data was collected (local time)                     |
| input_desc    | text             | One of two possible values: OS:NETWORK INTERFACE::BYTES RECEIVED/SEC::, OS:NETWORK INTERFACE::BYTES SENT/SEC::, describing the data collected for the rest of the row |
| nrs           | integer          | The number of samples collected on that day. Note: the DCA collector samples every five (5) seconds, so to estimate the total duration, multiple **nrs** by 5. |
| min_bytes_sec | double precision | The minimum bytes/sec measured during the sampling period    |
| avg_bytes_sec | double precision | The average (mean) bytes/second measured during the sampling period |
| max_bytes_sec | double precision | The maximum bytes/second measured during the sampling period |

### system_on_off_suspend_time_day

This table provides a summarization of client on time, off time, modern sleep time, and sleep time a system experiences on any given date. The time columns (on_time, off_time, mods_time, and sleep_time) are all given in seconds of time.

| Column     | Type      | Description                                                  |
| ---------- | --------- | ------------------------------------------------------------ |
| load_ts    | timestamp | Time stamp for when the data was loaded into the table       |
| guid       | text      | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt         | date      | Date the data was collected (local time)                     |
| on_time    | bigint    | The amount of time the client was on (active) on the date, in seconds |
| off_time   | bigint    | The amount of time the client was off on the date, in seconds |
| mods_time  | bigint    | The amount of time the client was in a modern sleep state, seconds |
| sleep_time | bigint    | The amount of time the client was in a sleep state on the date, in seconds |

### system_os_codename_history

This table provides a history of Windows operating system changes over time for a system. The **min_ts** and **max_ts** timestamp columns provide the time window for when a particular Windows version was running on the system.

| Column      | Type      | Description                                                  |
| ----------- | --------- | ------------------------------------------------------------ |
| load_ts     | timestamp | Time stamp for when the data was loaded into the table       |
| guid        | text      | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| min_ts      | timestamp | The first time the DCA collector detected this version of the operating system |
| max_ts      | timestamp | The last time the DCA collector detected this version of the operating system |
| os_name     | text      | The major Windows family designation for the Windows version. The allowed versions are:<br />Unknown - DCA was unable to determine the Os version. Probably a server version.<br/>Win10<br/>Win11<br/>Win7<br/>Win8<br/>Win8.1<br/>Win Vista |
| os_codename | text      | The codename for the specific Windows version (19H1, 22H2, RS1, etc.) |

### system_pkg_avg_freq_mhz

A dataset providing a statistical summarization of the average frequency (clock speed) of the client processor, at the whole processor level. The actual processor can vary its clock speed on a per core basis; this table summarizes it over all the cores to provide a holistic view.

There was little to no filtering performed on this table. It should be noted that any rows with a **nrs** value less than 6 (30 seconds) should probably be filtered out as unreliable because of limitations in the DCA collector.

| Column           | Type             | Description                                                  |
| ---------------- | ---------------- | ------------------------------------------------------------ |
| load_ts          | timestamp        | Time stamp for when the data was loaded into the table       |
| guid             | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt               | date             | Date the data was collected (local time)                     |
| event_name       | text             | Provides an enumeration of the system power state. It can be one of the following states:<br />AC_DISPLAY_OFF - On AC power, display is off<br/>AC_DISPLAY_ON - On AC power, display is on<br/>AC_DISPLAY_UN - On AC power, display state is unknown<br/>DC_DISPLAY_OFF - On DC power, display is off<br/>DC_DISPLAY_ON - On DC power, display is on<br/>DC_DISPLAY_UN - On DC power, display state is unknown<br/>UN_DISPLAY_OFF - Unknown power state, display is off<br/>UN_DISPLAY_ON - Unknown power state, display is on<br/>UN_DISPLAY_UN - Unknown power state, display state is unknown |
| nrs              | bigint           | The number of samples captured during this event. Note that the DCA collector collects every 5 seconds |
| number_of_cores  | smallint         | The number of processor cores                                |
| duration_ms      | bigint           | The duration of the event. Note that this duration should be within 10 seconds of the **nrs** * 5 |
| min_avg_freq_mhz | double precision | The minimum average frequency (clock rate) of the processor during these events over ALL cores on this date, in megahertz (MHz) |
| avg_avg_freq_mhz | double precision | The average of the average frequency (clock rate) of the processor during these events over ALL cores on this date, in megahertz (MHz) |
| med_avg_freq_mhz | double precision | The median of the average frequency (clock rate) of the processor during these events over ALL cores on this date, in megahertz (MHz) |
| max_avg_freq_mhz | double precision | The maximum of the average frequency (clock rate) of the processor during these events over ALL cores on this date, in megahertz (MHz) |

### system_pkg_c0

This dataset statistically characterizes the processor C0 state. The values are recorded as a percentage of time (0-100%). For reference,

In the C0 state:

- The processor is fully operational
- All processor functions are active
- Instructions are being executed
- The processor is consuming its maximum power

| Column          | Type             | Description                                                  |
| --------------- | ---------------- | ------------------------------------------------------------ |
| load_ts         | timestamp        | Time stamp for when the data was loaded into the table       |
| guid            | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt              | date             | Date the data was collected (local time)                     |
| event_name      | text             | Provides an enumeration of the system power state. It can be one of the following states:<br />AC_DISPLAY_OFF - On AC power, display is off<br/>AC_DISPLAY_ON - On AC power, display is on<br/>AC_DISPLAY_UN - On AC power, display state is unknown<br/>DC_DISPLAY_OFF - On DC power, display is off<br/>DC_DISPLAY_ON - On DC power, display is on<br/>DC_DISPLAY_UN - On DC power, display state is unknown<br/>UN_DISPLAY_OFF - Unknown power state, display is off<br/>UN_DISPLAY_ON - Unknown power state, display is on<br/>UN_DISPLAY_UN - Unknown power state, display state is unknown |
| nrs             | bigint           | The number of samples captured during this event. Note that the DCA collector collects every 5 seconds |
| number_of_cores | smallint         | The number of processor cores                                |
| duration_ms     | bigint           | The duration of the event. Note that this duration should be within 10 seconds of the **nrs** * 5 |
| min_pkg_c0      | double precision | The minimum package C0 percentage of the processor during these events over ALL cores on this date |
| avg_pkg_c0      | double precision | The average package C0 percentage of the processor during these events over ALL cores on this date |
| med_pkg_c0      | double precision | The median package C0 percentage of the processor during these events over ALL cores on this date |
| max_pkg_c0      | double precision | The maximum package C0 percentage of the processor during these events over ALL cores on this date |

### system_pkg_temp_centigrade

This dataset provides a processor level summarization of the temperatures the processor is generating during the particular events on a particular day.

| Column              | Type             | Description                                                  |
| ------------------- | ---------------- | ------------------------------------------------------------ |
| load_ts             | timestamp        | Time stamp for when the data was loaded into the table       |
| guid                | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                  | date             | Date the data was collected (local time)                     |
| event_name          | text             | Provides an enumeration of the system power state. It can be one of the following states:<br />AC_DISPLAY_OFF - On AC power, display is off<br/>AC_DISPLAY_ON - On AC power, display is on<br/>AC_DISPLAY_UN - On AC power, display state is unknown<br/>DC_DISPLAY_OFF - On DC power, display is off<br/>DC_DISPLAY_ON - On DC power, display is on<br/>DC_DISPLAY_UN - On DC power, display state is unknown<br/>UN_DISPLAY_OFF - Unknown power state, display is off<br/>UN_DISPLAY_ON - Unknown power state, display is on<br/>UN_DISPLAY_UN - Unknown power state, display state is unknown |
| nrs                 | bigint           | The number of samples captured during this event. Note that the DCA collector collects every 5 seconds |
| number_of_cores     | smallint         | The number of processor cores                                |
| duration_ms         | bigint           | The duration of the event. Note that this duration should be within 10 seconds of the **nrs** * 5 |
| min_temp_centigrade | double precision | The minimum package temperature (in degrees centigrade) of the processor during these events over ALL cores on this date |
| avg_temp_centigrade | double precision | The average package temperature (in degrees centigrade) of the processor during these events over ALL cores on this date |
| med_temp_centigrade | double precision | The median package temperature (in degrees centigrade) of the processor during these events over ALL cores on this date |
| max_temp_centigrade | double precision | The maximum package temperature (in degrees centigrade) of the processor during these events over ALL cores on this date |

### system_psys_rap_watts

This dataset provides an estimate/statistical summary of the total system power consumed by the client on the date for the various events it collected on. This is an underestimate, because it is not capturing some client component elements like fan power usage, etc. The values are in Watts.

| Column             | Type             | Description                                                  |
| ------------------ | ---------------- | ------------------------------------------------------------ |
| load_ts            | timestamp        | Time stamp for when the data was loaded into the table       |
| guid               | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                 | date             | Date the data was collected (local time)                     |
| event_name         | text             | Provides an enumeration of the system power state. It can be one of the following states:<br />AC_DISPLAY_OFF - On AC power, display is off<br/>AC_DISPLAY_ON - On AC power, display is on<br/>AC_DISPLAY_UN - On AC power, display state is unknown<br/>DC_DISPLAY_OFF - On DC power, display is off<br/>DC_DISPLAY_ON - On DC power, display is on<br/>DC_DISPLAY_UN - On DC power, display state is unknown<br/>UN_DISPLAY_OFF - Unknown power state, display is off<br/>UN_DISPLAY_ON - Unknown power state, display is on<br/>UN_DISPLAY_UN - Unknown power state, display state is unknown |
| nrs                | bigint           | The number of samples captured during this event. Note that the DCA collector collects every 5 seconds |
| duration_ms        | bigint           | The duration of the event. Note that this duration should be within 10 seconds of the **nrs** * 5 |
| min_psys_rap_watts | double precision | The minimum wattage (in Watts) of the client system during these events on this date |
| avg_psys_rap_watts | double precision | The average wattage (in Watts) of the client system during these events on this date |
| med_psys_rap_watts | double precision | The median wattage (in Watts) of the client system during these events on this date |
| max_psys_rap_watts | double precision | The maximum wattage (in Watts) of the client system during these events on this date |

### system_userwait

This dataset was part of the DCA collection used to measure user "frustration" by proxy of measuring the amount of time the user spends waiting for certain events to occur on their system. This dataset measures WAIT events for application starts and general WAITs when an application is busy. It also provides estimates for NON-WAIT events, as well as those WAIT events that were DISCARDED (due to shortness.)

| Column               | Type    | Description                                                  |
| -------------------- | ------- | ------------------------------------------------------------ |
| guid                 | text    | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                   | date    | Date the data was collected (local time)                     |
| event_name           | text    | Provides an enumeration of the different events this dataset collects. They are:<br />WAIT<br/>TOTAL_NON_WAIT_EVENTS<br/>APPSTARTING<br/>TOTAL_DISCARDED_WAIT_EVENTS<br />Records either WAIT (spinning wheel incident) or APPSTARTING currently (spinning wheel incident occurred when trying to bring another application into focus or launch new app). Custom events added: 1) TOTAL_DISCARDED_WAIT_EVENTS (Records the total amount of wait event we discarded because they were below the threshold) and 2) TOTAL_NON_WAIT_EVENTS (records how many events we discarded because they had different event statuses) Note: these custom events do not have associated foreground data (see default values for them listed in this case) |
| ac_dc_event_name     | text    | Provides an enumeration of the system power state. It can be one of the following states:<br />AC_DISPLAY_OFF - On AC power, display is off<br/>AC_DISPLAY_ON - On AC power, display is on<br/>AC_DISPLAY_UN - On AC power, display state is unknown<br/>DC_DISPLAY_OFF - On DC power, display is off<br/>DC_DISPLAY_ON - On DC power, display is on<br/>DC_DISPLAY_UN - On DC power, display state is unknown<br/>UN_DISPLAY_OFF - Unknown power state, display is off<br/>UN_DISPLAY_ON - Unknown power state, display is on<br/>UN_DISPLAY_UN - Unknown power state, display state is unknown<br />There are two extra enumeration values in here that were NOT in the **event_name** columns in  **system_psys_rap_watts** and alike. They are<br />NA - Not applicable, and<br />unknown - unknown - self explanatory. |
| proc_name            | text    | The application process name that the wait state. In the case for the custom wait states described in the **event_name** column above for this dataset, this will be set to NA. |
| number_of_instances  | integer | The number of instances measured                             |
| total_duration_in_ms | bigint  | The total wait time in milliseconds (ms)                     |

### system_web_cat_pivot

This dataset provides a global overview of the web usage for client systems. There is no clear indication of the metric collected, though it is almost certainly duration in milliseconds. It summarizes this value over the different web categories the DCA collector bins web usage as, with nulls for any category that has NO value.

| Column                                     | Type             | Description                                                  |
| ------------------------------------------ | ---------------- | ------------------------------------------------------------ |
| guid                                       | text             | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| social_social_network                      | double precision | Total amount of time spent on social networks (Facebook, Instagram, etc.) |
| private_private                            | double precision | Time spent web browsing in a private mode                    |
| productivity_word_processing               | double precision | Time spent word processing one the web (Google Docs, etc.)   |
| news_news                                  | double precision | Time spent on news sites/portals                             |
| social_communication                       | double precision | Time spent on social communication websites (Communicator, etc.) |
| productivity_spreadsheets                  | double precision | Time spent on web-based spreadsheets (Google Sheets, etc.)   |
| content_creation_photo_edit_creation       | double precision | Time spent on web-based photo-editing sites                  |
| search_search                              | double precision | Time spent on web search (Google, etc.)                      |
| productivity_other                         | double precision | Time spent on office based productivity web sites            |
| entertainment_music_audio_streaming        | double precision | Time spent on music streaming (Spotify, Amazon Music, etc.)  |
| finance_banking_and_accounting             | double precision | Time spent performing online banking tasks, etc.             |
| games_other                                | double precision | Time spent on gaming sites, not necessarily playing games    |
| entertainment_other                        | double precision | Time spent on entertainment web sites, not categorized by the other entertainment sub-categories |
| education_education                        | double precision | Time spent on education web sites (Khan Academy, etc.)       |
| productivity_programming                   | double precision | Time spent on programming web sites                          |
| productivity_presentations                 | double precision | Time spent on presentation web sites (Google Slides, etc.)   |
| reference_reference                        | double precision | Time spent on reference web sites (wikipedia, etc.)          |
| shopping_shopping                          | double precision | Time spent on shopping web sites (Amazon, etc.)              |
| other_unclassified                         | double precision | Time spent on web sites for which the DCA collector either cannot or will not classify the web sites visited (adult web sites, etc.) |
| recreational_travel                        | double precision | Time spent on travel web sites (Expedia, etc.)               |
| entertainment_video_streaming              | double precision | Time spent on video streaming web sites (youtube, etc.)      |
| games_video_games                          | double precision | Time spent on video game web sites                           |
| productivity_crm                           | double precision | Time spent on customer relationship management web sites     |
| mail_mail                                  | double precision | Time browsing email (Gmail, Yahoo mail, etc.)                |
| social_communication_live                  | double precision | Time spent on social networking in a live context (Zoom, etc.) |
| content_creation_video_audio_edit_creation | double precision | Time spent performing audio/video editing/creation           |
| productivity_project_management            | double precision | Time spent on project management web sites (agile development sites, etc.) |
| content_creation_web_design_development    | double precision | Time spent doing web design and development in a browser     |

### system_web_cat_pivot_domain_count

This is a dataset that characterized the web usage of client systems based on the domain count metric collected in the **web_cat_usage_v2** dataset. Unlike the **system_web_cat_pivot** dataset, this is provided at a **guid, dt** level.

| Column                                     | Type   | Description                                                  |
| ------------------------------------------ | ------ | ------------------------------------------------------------ |
| guid                                       | text   | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                                         | date   | Date the data was collected (local time)                     |
| content_creation_photo_edit_creation       | bigint | Domain count of web-based photo-editing sites                |
| content_creation_video_audio_edit_creation | bigint | Domain count of audio/video editing/creation web sites       |
| content_creation_web_design_development    | bigint | Domain count of web design and development web sites         |
| education                                  | bigint | Domain count of educational web sites                        |
| entertainment_music_audio_streaming        | bigint | Domain count of music streaming web sites (spotify, etc.)    |
| entertainment_other                        | bigint | Domain count of uncharacterized entertainment web sites      |
| entertainment_video_streaming              | bigint | Domain count of video streaming web sites (youtube, etc.)    |
| finance                                    | bigint | Domain count of finance web sites (yahoo finance, etc.)      |
| games_other                                | bigint | Domain count of otherwise uncharacterized gaming web sites   |
| games_video_games                          | bigint | Domain count of video game web sites                         |
| mail                                       | bigint | Domain count of email web sites (yahoo mail, Gmail, etc.)    |
| news                                       | bigint | Domain count of news web sites                               |
| unclassified                               | bigint | Domain count of unclassified web sites (adult web sites, unknowns, etc.) |
| private                                    | bigint | Domain count of web sites browsed in private mode            |
| productivity_crm                           | bigint | Domain count of web sites for customer relationship management |
| productivity_other                         | bigint | Domain count of productivity web sites not otherwise categorized |
| productivity_presentations                 | bigint | Domain count of productivity web sites for making presentations (Google Slides, etc.) |
| productivity_programming                   | bigint | Domain count of productivity web sites for programming       |
| productivity_project_management            | bigint | Domain count of web sites for project management             |
| productivity_spreadsheets                  | bigint | Domain count of web sites for spreadsheets (Google Sheets, etc.) |
| productivity_word_processing               | bigint | Domain count of word processing web sites (Google Docs, etc.) |
| recreation_travel                          | bigint | Domain count recreational travel web sites (Expedia, hotels.com, etc.) |
| reference                                  | bigint | Domain count for reference web sites (wikipedia, etc.)       |
| search                                     | bigint | Domain count for search engines (Google, Yahoo, etc.)        |
| shopping                                   | bigint | Domain count for shopping web sites (Amazon, etc.)           |
| social_social_network                      | bigint | Domain count for social networking web sites (Facebook, Instagram, etc.) |
| social_communication                       | bigint | Domain count for social communication web sites (Twitter, etc.) |
| social_communication_live                  | bigint | Domain count for live social communication web sites (Zoom, Skype, etc.) |

### system_web_cat_pivot_duration

This is a dataset that characterized the web usage of client systems based on the duration metric collected in the **web_cat_usage_v2** dataset. Unlike the **system_web_cat_pivot** dataset, this is provided at a **guid, dt** level. The units are in milliseconds (ms).

| Column                                     | Type   | Description                                                  |
| ------------------------------------------ | ------ | ------------------------------------------------------------ |
| guid                                       | text   | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                                         | date   | Date the data was collected (local time)                     |
| content_creation_photo_edit_creation       | bigint | Sum duration of web-based photo-editing sites                |
| content_creation_video_audio_edit_creation | bigint | Sum duration of audio/video editing/creation web sites       |
| content_creation_web_design_development    | bigint | Sum duration of web design and development web sites         |
| education                                  | bigint | Sum duration of educational web sites                        |
| entertainment_music_audio_streaming        | bigint | Sum duration of music streaming web sites (spotify, etc.)    |
| entertainment_other                        | bigint | Sum duration of uncharacterized entertainment web sites      |
| entertainment_video_streaming              | bigint | Sum duration of video streaming web sites (youtube, etc.)    |
| finance                                    | bigint | Sum duration of finance web sites (yahoo finance, etc.)      |
| games_other                                | bigint | Sum duration of otherwise uncharacterized gaming web sites   |
| games_video_games                          | bigint | Sum duration of video game web sites                         |
| mail                                       | bigint | Sum duration of email web sites (yahoo mail, Gmail, etc.)    |
| news                                       | bigint | Sum duration of news web sites                               |
| unclassified                               | bigint | Sum duration of unclassified web sites (adult web sites, unknowns, etc.) |
| private                                    | bigint | Sum duration of web sites browsed in private mode            |
| productivity_crm                           | bigint | Sum duration of web sites for customer relationship management |
| productivity_other                         | bigint | Sum duration of productivity web sites not otherwise categorized |
| productivity_presentations                 | bigint | Sum duration of productivity web sites for making presentations (Google Slides, etc.) |
| productivity_programming                   | bigint | Sum duration of productivity web sites for programming       |
| productivity_project_management            | bigint | Sum duration of web sites for project management             |
| productivity_spreadsheets                  | bigint | Sum duration of web sites for spreadsheets (Google Sheets, etc.) |
| productivity_word_processing               | bigint | Sum duration of word processing web sites (Google Docs, etc.) |
| recreation_travel                          | bigint | Sum duration recreational travel web sites (Expedia, hotels.com, etc.) |
| reference                                  | bigint | Sum duration for reference web sites (wikipedia, etc.)       |
| search                                     | bigint | Sum duration for search engines (Google, Yahoo, etc.)        |
| shopping                                   | bigint | Sum duration for shopping web sites (Amazon, etc.)           |
| social_social_network                      | bigint | Sum duration for social networking web sites (Facebook, Instagram, etc.) |
| social_communication                       | bigint | Sum duration for social communication web sites (Twitter, etc.) |
| social_communication_live                  | bigint | Sum duration for live social communication web sites (Zoom, Skype, etc.) |

### system_web_cat_pivot_page_load_count

This is a dataset that characterized the web usage of client systems based on the page load count metric collected in the **web_cat_usage_v2** dataset. Unlike the **system_web_cat_pivot** dataset, this is provided at a **guid, dt** level. 

| Column                                     | Type   | Description                                                  |
| ------------------------------------------ | ------ | ------------------------------------------------------------ |
| guid                                       | text   | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                                         | date   | Date the data was collected (local time)                     |
| content_creation_photo_edit_creation       | bigint | Page load count of web-based photo-editing sites             |
| content_creation_video_audio_edit_creation | bigint | Page load count of audio/video editing/creation web sites    |
| content_creation_web_design_development    | bigint | Page load count of web design and development web sites      |
| education                                  | bigint | Page load count of educational web sites                     |
| entertainment_music_audio_streaming        | bigint | Page load count of music streaming web sites (spotify, etc.) |
| entertainment_other                        | bigint | Page load count of uncharacterized entertainment web sites   |
| entertainment_video_streaming              | bigint | Page load count of video streaming web sites (youtube, etc.) |
| finance                                    | bigint | Page load count of finance web sites (yahoo finance, etc.)   |
| games_other                                | bigint | Page load count of otherwise uncharacterized gaming web sites |
| games_video_games                          | bigint | Page load count of video game web sites                      |
| mail                                       | bigint | Page load count of email web sites (yahoo mail, Gmail, etc.) |
| news                                       | bigint | Page load count of news web sites                            |
| unclassified                               | bigint | Page load count of unclassified web sites (adult web sites, unknowns, etc.) |
| private                                    | bigint | Page load count of web sites browsed in private mode         |
| productivity_crm                           | bigint | Page load count of web sites for customer relationship management |
| productivity_other                         | bigint | Page load count of productivity web sites not otherwise categorized |
| productivity_presentations                 | bigint | Page load count of productivity web sites for making presentations (Google Slides, etc.) |
| productivity_programming                   | bigint | Page load count of productivity web sites for programming    |
| productivity_project_management            | bigint | Page load count of web sites for project management          |
| productivity_spreadsheets                  | bigint | Page load count of web sites for spreadsheets (Google Sheets, etc.) |
| productivity_word_processing               | bigint | Page load count of word processing web sites (Google Docs, etc.) |
| recreation_travel                          | bigint | Page load count recreational travel web sites (Expedia, hotels.com, etc.) |
| reference                                  | bigint | Page load count for reference web sites (wikipedia, etc.)    |
| search                                     | bigint | Page load count for search engines (Google, Yahoo, etc.)     |
| shopping                                   | bigint | Page load count for shopping web sites (Amazon, etc.)        |
| social_social_network                      | bigint | Page load count for social networking web sites (Facebook, Instagram, etc.) |
| social_communication                       | bigint | Page load count for social communication web sites (Twitter, etc.) |
| social_communication_live                  | bigint | Page load count for live social communication web sites (Zoom, Skype, etc.) |

### system_web_cat_pivot_page_visit_count

This is a dataset that characterized the web usage of client systems based on the page visit count metric collected in the **web_cat_usage_v2** dataset. Unlike the **system_web_cat_pivot** dataset, this is provided at a **guid, dt** level.

| Column                                     | Type   | Description                                                  |
| ------------------------------------------ | ------ | ------------------------------------------------------------ |
| guid                                       | text   | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                                         | date   | Date the data was collected (local time)                     |
| content_creation_photo_edit_creation       | bigint | Page visit count of web-based photo-editing sites            |
| content_creation_video_audio_edit_creation | bigint | Page visit count of audio/video editing/creation web sites   |
| content_creation_web_design_development    | bigint | Page visit count of web design and development web sites     |
| education                                  | bigint | Page visit count of educational web sites                    |
| entertainment_music_audio_streaming        | bigint | Page visit count of music streaming web sites (spotify, etc.) |
| entertainment_other                        | bigint | Page visit count of uncharacterized entertainment web sites  |
| entertainment_video_streaming              | bigint | Page visit count of video streaming web sites (youtube, etc.) |
| finance                                    | bigint | Page visit count of finance web sites (yahoo finance, etc.)  |
| games_other                                | bigint | Page visit count of otherwise uncharacterized gaming web sites |
| games_video_games                          | bigint | Page visit count of video game web sites                     |
| mail                                       | bigint | Page visit count of email web sites (yahoo mail, Gmail, etc.) |
| news                                       | bigint | Page visit count of news web sites                           |
| unclassified                               | bigint | Page visit count of unclassified web sites (adult web sites, unknowns, etc.) |
| private                                    | bigint | Page visit count of web sites browsed in private mode        |
| productivity_crm                           | bigint | Page visit count of web sites for customer relationship management |
| productivity_other                         | bigint | Page visit count of productivity web sites not otherwise categorized |
| productivity_presentations                 | bigint | Page visit count of productivity web sites for making presentations (Google Slides, etc.) |
| productivity_programming                   | bigint | Page visit count of productivity web sites for programming   |
| productivity_project_management            | bigint | Page visit count of web sites for project management         |
| productivity_spreadsheets                  | bigint | Page visit count of web sites for spreadsheets (Google Sheets, etc.) |
| productivity_word_processing               | bigint | Page visit count of word processing web sites (Google Docs, etc.) |
| recreation_travel                          | bigint | Page visit count recreational travel web sites (Expedia, hotels.com, etc.) |
| reference                                  | bigint | Page visit count for reference web sites (wikipedia, etc.)   |
| search                                     | bigint | Page visit count for search engines (Google, Yahoo, etc.)    |
| shopping                                   | bigint | Page visit count for shopping web sites (Amazon, etc.)       |
| social_social_network                      | bigint | Page visit count for social networking web sites (Facebook, Instagram, etc.) |
| social_communication                       | bigint | Page visit count for social communication web sites (Twitter, etc.) |
| social_communication_live                  | bigint | Page visit count for live social communication web sites (Zoom, Skype, etc.) |

### system_web_cat_pivot_site_count

This is a dataset that characterized the web usage of client systems based on the site count metric collected in the **web_cat_usage_v2** dataset. Unlike the **system_web_cat_pivot** dataset, this is provided at a **guid, dt** level.

| Column                                     | Type   | Description                                                  |
| ------------------------------------------ | ------ | ------------------------------------------------------------ |
| guid                                       | text   | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt                                         | date   | Date the data was collected (local time)                     |
| content_creation_photo_edit_creation       | bigint | Site count of web-based photo-editing sites                  |
| content_creation_video_audio_edit_creation | bigint | Site count of audio/video editing/creation web sites         |
| content_creation_web_design_development    | bigint | Site count of web design and development web sites           |
| education                                  | bigint | Site count of educational web sites                          |
| entertainment_music_audio_streaming        | bigint | Site count of music streaming web sites (spotify, etc.)      |
| entertainment_other                        | bigint | Site count of uncharacterized entertainment web sites        |
| entertainment_video_streaming              | bigint | Site count of video streaming web sites (youtube, etc.)      |
| finance                                    | bigint | Site count of finance web sites (yahoo finance, etc.)        |
| games_other                                | bigint | Site count of otherwise uncharacterized gaming web sites     |
| games_video_games                          | bigint | Site count of video game web sites                           |
| mail                                       | bigint | Site count of email web sites (yahoo mail, Gmail, etc.)      |
| news                                       | bigint | Site count of news web sites                                 |
| unclassified                               | bigint | Site count of unclassified web sites (adult web sites, unknowns, etc.) |
| private                                    | bigint | Site count of web sites browsed in private mode              |
| productivity_crm                           | bigint | Site count of web sites for customer relationship management |
| productivity_other                         | bigint | Site count of productivity web sites not otherwise categorized |
| productivity_presentations                 | bigint | Site count of productivity web sites for making presentations (Google Slides, etc.) |
| productivity_programming                   | bigint | Site count of productivity web sites for programming         |
| productivity_project_management            | bigint | Site count of web sites for project management               |
| productivity_spreadsheets                  | bigint | Site count of web sites for spreadsheets (Google Sheets, etc.) |
| productivity_word_processing               | bigint | Site count of word processing web sites (Google Docs, etc.)  |
| recreation_travel                          | bigint | Site count recreational travel web sites (Expedia, hotels.com, etc.) |
| reference                                  | bigint | Site count for reference web sites (wikipedia, etc.)         |
| search                                     | bigint | Site count for search engines (Google, Yahoo, etc.)          |
| shopping                                   | bigint | Site count for shopping web sites (Amazon, etc.)             |
| social_social_network                      | bigint | Site count for social networking web sites (Facebook, Instagram, etc.) |
| social_communication                       | bigint | Site count for social communication web sites (Twitter, etc.) |
| social_communication_live                  | bigint | Site count for live social communication web sites (Zoom, Skype, etc.) |

### system_web_cat_usage

This is a date level summary statistics table of the web usage of the client systems. It provides the browser that was used (Chrome, Edge, and Firefox are supported), the web category of the browsing, and various metrics about that browsing. The other web category tables above do something similar, but broken out by the metrics, and pivoted by the categories. This is provided to support alternate data mining queries.

| Column           | Type      | Description                                                  |
| ---------------- | --------- | ------------------------------------------------------------ |
| load_ts          | timestamp | Time stamp for when the data was loaded into the table       |
| guid             | text      | Globally Unique IDentifier assigned to each and every client system in the DCA dataset |
| dt               | date      | Date the data was collected (local time)                     |
| browser          | text      | One of three supported browsers: chrome, edge, or firefox.   |
| parent_category  | text      | One of the following supported categories:<br />content creation<br/>education<br/>entertainment<br/>finance<br/>games<br/>mail<br/>news<br/>other<br/>private<br/>productivity<br/>recreation<br/>reference<br/>search<br/>shopping<br/>social |
| sub_category     | text      | A column providing a finer "granularity" to the parent_category. For example, parent category **social** contains three sub_categories: **social network, communication, and communication - live**. |
| duration_ms      | bigint    | The sum duration in milliseconds (ms) spent on this **browser** on this **parent_category, sub_category** web browsing categorization. |
| page_load_count  | integer   | The page load count spent on this **browser** on this **parent_category, sub_category** web browsing categorization. |
| site_count       | integer   | The site count spent on this **browser** on this **parent_category, sub_category** web browsing categorization. |
| domain_count     | integer   | The domain count spent on this **browser** on this **parent_category, sub_category** web browsing categorization. |
| page_visit_count | integer   | The page visit count spent on this **browser** on this **parent_category, sub_category** web browsing categorization. |

