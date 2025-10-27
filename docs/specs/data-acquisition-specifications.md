# DataQ DI-2108 Data Acquisition System - Project Specifications

## Project Overview

A Python-based data acquisition application for the DataQ DI-2108 device that captures analog signals from multiple channels, displays them in real-time, and saves the data to CSV files with comprehensive metadata.

## Reference Documentation

- DataQ DI-2108 example Python file (`.py-example`)
- DataQ protocol specification (PDF)

## System Requirements

### Hardware
- DataQ DI-2108 data acquisition device
- USB connection to host computer
- Analog signal sources connected to device channels

### Software
- Python 3.x
- Required libraries for serial communication, data visualization, and GUI

## Default Configuration Values

The application shall initialize with the following default values on launch:

| Parameter | Default Value | Unit |
|-----------|---------------|------|
| Number of Channels | 8 | - |
| Channel Names | A0, A1, A2, A3, A4, A5, A6, A7 | - |
| Channel Units | V | volts |
| Display Sampling Rate | 20 | Hz |
| Acquisition Sampling Rate | 5000 | Hz |
| X-Position | 0 | (user-defined units) |
| Y-Position | 0 | (user-defined units) |
| Z-Position | 0 | (user-defined units) |

**Note**: Position values persist between acquisitions and only reset to defaults on application restart.

## Functional Requirements

### 1. Configuration Parameters

The application shall provide a GUI with configuration controls that remain accessible throughout the application lifecycle:

#### Channel Configuration
- **Number of Active Channels**: Selectable number of channels to acquire (within device limits)
  - **Default**: 8 channels
- **Channel Headers**: Custom text labels for each active channel
  - **Default**: A0, A1, A2, A3, A4, A5, A6, A7
- **Channel Units**: Display units for measurements
  - **Default**: V (volts) for all channels

#### Acquisition Parameters
- **Acquisition Sampling Rate**: Rate at which data is captured during acquisition (Hz)
  - **Default**: 5000 Hz (5 kHz)
  - Applies to all channels during acquisition
- **Acquisition Duration**: Total time length for data capture (seconds or minutes)
- **Measurement Direction**: Choice between "X" or "Y" axis orientation
  - Selectable via radio buttons or dropdown
- **Spatial Position**: Three-dimensional coordinates of the measurement location
  - X-position input field (numeric value with units)
  - Y-position input field (numeric value with units)
  - Z-position input field (numeric value with units)
  - **Default on Launch**: X=0, Y=0, Z=0
  - **Persistence**: Values remain set between acquisitions until manually changed

#### Display Parameters
- **Display Sampling Rate**: Refresh rate for live visualization (Hz)
  - **Default**: 20 Hz
  - Independent from acquisition sampling rate
  - Applies to the live display both before and during acquisition

### 2. User Interface Requirements

#### Main Application Window
The application shall present a unified GUI containing:

**Control Panel Section:**
- Channel configuration controls (number of channels, headers)
- Acquisition sampling rate input field (default: 5000 Hz)
- Display sampling rate input field (default: 20 Hz)
- Acquisition duration input field
- Measurement direction selector (X or Y)
- Position input fields:
  - X-position (default: 0)
  - Y-position (default: 0)
  - Z-position (default: 0)
- **Save Parameters Button**: Applies configuration changes and reconfigures device
- **Start Acquisition Button**: Begins saving data to file at the current acquisition sampling rate
- Status indicator showing current state (idle, acquiring, processing)

**Live Display Section:**
- Real-time plot area showing all active channels
- Updates continuously at the display sampling rate
- Active from application launch
- Shows downsampled data from the continuous high-speed acquisition

#### Application Launch Behavior
- Upon startup, the application shall:
  - Initialize connection with DataQ DI-2108 device
  - Configure device with default acquisition sampling rate (5000 Hz)
  - Begin streaming data continuously at the acquisition sampling rate
  - Downsample data in software for display at display sampling rate (20 Hz default)
  - Display live signals immediately in the plot area
  - Show default configuration values in all control fields
  - Display "Ready" or "Idle" status

#### Save Parameters Button Behavior
- Applies any changes to channel configuration, acquisition rate, or other device parameters
- Reconfigures the DataQ DI-2108 device with new settings
- Continuous data streaming and live display continue without interruption at the new acquisition rate
- Only time device reconfiguration occurs

#### Start Acquisition Button Behavior
- When clicked, begins saving incoming data to memory buffer
- Does NOT reconfigure the device (device already streaming at acquisition rate)
- Button label changes to indicate active acquisition (e.g., "Stop Acquisition" or disabled state)
- Live display continues updating during data saving
- After acquisition duration completes, stops saving data and exports to CSV file
- Returns to idle state with live display continuing

### 3. Data Acquisition

#### Continuous High-Speed Acquisition (Always Active)
- Establish communication with DataQ DI-2108 device on application launch
- Configure device for continuous streaming at the acquisition sampling rate (default: 5000 Hz)
- Acquire analog signals from all active channels continuously
- Maintain precise timing according to configured acquisition sampling rate
- Continuous operation from launch until application close
- Data streams continuously but is only saved to file when Start button is pressed

#### Display Downsampling
- Downsample the continuous high-speed data stream in software for display purposes
- Display refresh rate determined by display sampling rate setting (default: 20 Hz)
- Downsampling factor: acquisition rate / display rate (e.g., show every 250th sample at defaults)
- Live display shows representative subset of acquired data
- Full-rate data always available for saving when acquisition triggered

#### Triggered Data Saving (Start Button)
- When Start button pressed, begin storing incoming samples to memory buffer
- Save all samples at full acquisition sampling rate
- Continue for the specified acquisition duration
- No device reconfiguration needed (already acquiring at correct rate)
- After duration expires, stop saving and export buffer to CSV file

#### Device Reconfiguration (Save Parameters Button)
- Only occurs when Save Parameters button is pressed
- Applies changes to:
  - Number of channels
  - Acquisition sampling rate
  - Channel assignments
- Reconfigures DataQ DI-2108 device with new settings
- Continuous data streaming resumes at new rate after brief reconfiguration

#### Data Management
- Circular or rolling buffer for display data (recent samples only)
- Dedicated buffer for acquisition data (all samples during triggered save)
- Maintain channel synchronization throughout operation
- Handle continuous data stream from device without data loss
- Clear acquisition buffer after successful CSV export

### 4. Real-Time Visualization

#### Live Display Window
- Active from application launch and continuously running
- Embedded in main application window
- Display all active channels in real-time
- Update plots at the configured display sampling rate (default: 20 Hz)
- Shows downsampled data from continuous high-speed acquisition stream
- Display recent time window (e.g., last 10-30 seconds of data)
- Scrolling time axis to show most recent data

#### Display Throughout Operation
- **Always shows downsampled view** of the continuous acquisition at acquisition sampling rate
- Provides visual confirmation of signal sources and device operation
- Allows user to verify connections and signal quality at all times
- No change in display behavior when Start button pressed (only triggers data saving)

#### Downsampling Display
- Software decimation of high-speed data for visualization
- Display every Nth sample where N = (acquisition rate / display rate)
- Example: At 5000 Hz acquisition and 20 Hz display, show every 250th sample
- Maintains time synchronization between actual acquisition and display

#### Display Features
- Clear channel identification using configured headers
- Appropriate axis labels with units (V for volts)
- Time axis showing elapsed time
- Grid lines for readability
- Auto-scaling y-axis or fixed scale option for each channel
- Legend identifying each channel with its header
- Visual indicator when data is being saved (e.g., colored border, "Recording" status)

### 5. Post-Acquisition Analysis

#### Data Visualization
After acquisition completes, present a comprehensive plot window with:
- All acquired channels displayed in separate subplots or on the same axes
- Full resolution data (no downsampling)
- Time axis in appropriate units (seconds, minutes)

#### Synchronized Zoom Feature
- **Critical Requirement**: All channel plots shall share synchronized x-axes
- When user zooms into a time region on any plot, all other channel plots shall zoom to the same time range
- Pan operations shall also be synchronized across all plots
- Independent y-axis scaling for each channel

#### Interactive Features
- Zoom in/out capabilities
- Pan left/right through time series
- Grid lines for readability
- Legend identifying each channel

### 6. Data Export

#### CSV File Format
Save acquired data to a comma-separated values file with:
- Header row containing channel names as configured
- Subsequent rows with time-aligned sample values
- First column as time vector (seconds from start)
- Remaining columns for each channel's data
- Consistent decimal precision

#### File Naming Convention
Automatically generate filename using the pattern:

```
yyyy-mm-dd_hh-mm-ss_X[x-position]_Y[y-position]_Z[z-position]_[direction].csv
```

Where:
- `yyyy-mm-dd`: Acquisition date (year-month-day)
- `hh-mm-ss`: Acquisition start time (hour-minute-second, 24-hour format)
- `X[x-position]`: Literal "X" followed by the X-coordinate value
- `Y[y-position]`: Literal "Y" followed by the Y-coordinate value
- `Z[z-position]`: Literal "Z" followed by the Z-coordinate value
- `[direction]`: Either "x" or "y" based on user selection

**Example filename:**
```
2025-10-20_14-30-45_X10.5_Y20.3_Z5.0_x.csv
```

#### File Location
- Save files to a designated output directory
- Create directory if it doesn't exist
- Provide user feedback on successful file save with full path

### 7. Error Handling

The application shall handle:
- Device connection failures
- Invalid configuration parameters
- Data acquisition interruptions
- File system errors during save operations
- User cancellation of acquisition

Provide clear error messages and allow graceful recovery where possible.

## Technical Considerations

### Continuous Acquisition Architecture
The application operates with continuous high-speed data acquisition:

**Single Acquisition Mode**
- Device always streams at the configured acquisition sampling rate (default 5000 Hz)
- No mode switching between idle and acquisition states
- Data is continuously acquired from device
- Start button controls data saving to file, not device operation
- Save Parameters button is the only trigger for device reconfiguration

### Data Flow
1. **Device Stream**: Continuous data at acquisition rate (e.g., 5000 Hz)
2. **Software Downsampling**: Decimate stream for display (e.g., every 250th sample for 20 Hz display)
3. **Display Update**: Render downsampled data at display rate
4. **Conditional Saving**: When Start pressed, buffer all incoming samples at full rate

### Sampling Rate Management
- **Acquisition Sampling Rate**: Device data rate, always active (default: 5000 Hz)
- **Display Sampling Rate**: GUI refresh rate via software downsampling (default: 20 Hz)
- **Downsampling Ratio**: Calculated as (acquisition rate / display rate)
- Display rate should be lower than acquisition rate for efficient visualization
- Changing acquisition rate requires pressing Save Parameters button
- Changing display rate only affects visualization, no device reconfiguration needed

### Device Reconfiguration Triggers
Device reconfiguration occurs ONLY when "Save Parameters" button is pressed after changing:
- Number of active channels
- Acquisition sampling rate
- Channel assignments or configuration

Device reconfiguration is NOT needed for:
- Starting/stopping data saving
- Changing display sampling rate (software-only change)
- Updating position values
- Changing measurement direction
- Changing acquisition duration

### Device Communication
- Follow DataQ DI-2108 protocol specification precisely
- Handle serial communication timing requirements
- Implement proper device initialization and shutdown sequences
- Maintain continuous data stream during normal operation
- Brief interruption during reconfiguration (Save Parameters button)
- Reference the provided example Python file for protocol implementation patterns

### Performance Requirements
- Minimize latency in continuous data stream processing
- Ensure acquisition timing accuracy at configured rate (5000 Hz default)
- Handle sustained high-speed data rates without buffer overflow
- Efficient software downsampling algorithm for display
- Optimize memory usage for continuous operation and triggered buffering
- Maintain responsive GUI during continuous acquisition
- Fast buffer allocation when Start button pressed

### Data Integrity
- Maintain sample alignment across all channels
- Preserve timestamp accuracy throughout continuous operation
- Prevent data loss during continuous streaming
- Accurate downsampling that preserves signal characteristics for display
- Validate data before saving to file
- Separate management of display buffer (rolling/circular) and acquisition buffer (fixed duration)

## Application Workflow

### Startup Sequence
1. **Application Launch**: Start application
2. **Device Connection**: Establish communication with DataQ DI-2108
3. **Default Configuration Load**: Set all parameters to defaults
   - 8 channels (A0-A7)
   - Display sampling rate: 20 Hz
   - Acquisition sampling rate: 5000 Hz
   - Position: X=0, Y=0, Z=0
   - Units: V (volts)
4. **Device Configuration**: Configure device for continuous acquisition at 5000 Hz
5. **Continuous Acquisition Start**: Device begins streaming data at acquisition rate
6. **Live Display Start**: Begin displaying downsampled data at display rate
7. **Ready State**: Application ready for parameter changes or data saving

### Parameter Configuration (As Needed)
1. User adjusts parameters in GUI (channels, rates, etc.)
2. User clicks "Save Parameters" button
3. Device reconfigures with new settings
4. Continuous acquisition resumes at new acquisition rate
5. Live display updates with new downsampling ratio
6. Application returns to ready state

### Position Update (Frequent)
1. User updates X, Y, Z position values in GUI
2. User selects or changes measurement direction (X or Y)
3. No device reconfiguration needed
4. Values stored for next acquisition filename

### Data Saving Sequence (Repeatable)
1. **Pre-Check**: User verifies signal quality via live display
2. **Start Trigger**: User clicks "Start Acquisition" button
3. **Begin Recording**: Application starts storing incoming data to buffer
4. **Visual Feedback**: Display shows "Recording" indicator
5. **Live Display Continue**: Display continues showing downsampled real-time data
6. **Duration Timer**: Count down or up to specified acquisition duration
7. **Stop Recording**: After duration expires, stop saving data to buffer
8. **Post-Processing**: Generate comprehensive plots with synchronized axes from saved data
9. **Data Export**: Save buffered data to CSV file with generated filename
10. **Ready for Next**: Live display continues; position values persist; ready for next acquisition

### Shutdown
1. User closes application
2. Stop continuous data acquisition
3. Close device connection gracefully
4. Exit application

## User Experience Goals

- **Immediate Visual Feedback**: Live display active from launch showing full-rate acquisition (downsampled for display)
- **Continuous High-Speed Monitoring**: Always acquiring at full speed, ready to save at any moment
- **Intuitive Single-Window Interface**: All controls and visualization in one unified window
- **Explicit Parameter Application**: Clear "Save Parameters" button to apply configuration changes
- **Persistent Configuration**: Position values and settings remain between acquisitions
- **Simple Recording Workflow**: Start button immediately begins saving already-acquired data
- **No Reconfiguration Delays**: Starting acquisition is instant (no device reconfiguration needed)
- **Quick Repeated Measurements**: Easy to perform multiple acquisitions at the same position
- **Professional Visualization**: High-quality real-time and post-acquisition plots
- **Reliable Data Capture**: Consistent file organization and data integrity
- **Minimal Setup Time**: Sensible defaults allow immediate use for standard measurements
- **Efficient Operation**: Single continuous acquisition mode, no mode switching

## Future Extensibility Considerations

Design should accommodate potential future enhancements:
- Multiple measurement sessions without restart
- Data format export options beyond CSV
- Advanced signal processing or filtering
- Configurable trigger conditions
- Real-time data analysis and statistics display
- Batch acquisition with parameter sets