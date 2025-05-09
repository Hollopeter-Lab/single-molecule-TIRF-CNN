% FARRED_label_training_data.m
% UI to browse and annotate Green traces that are colocalized with Far-Red spots
% Supports tagging traces with step counts (keyboard and numpad), exporting to Excel, and loading multiple files

function coloc_green_viewer()
    % Select multiple .mat files
    [files, path] = uigetfile('*.mat', 'Select one or more .mat files', 'MultiSelect', 'on');
    if isequal(files, 0)
        disp('No file selected');
        return;
    end
    if ischar(files)
        files = {files};
    end

    allTraces = struct('trace', {}, 'stepCount', {}, 'sourceFile', {});

    for f = 1:length(files)
        data = load(fullfile(path, files{f}));

        if ~isfield(data, 'gridData')
            warning('File %s does not contain gridData. Skipping...', files{f});
            continue;
        end

        gridData = data.gridData;

        % Loop over grid positions and extract colocGreen traces
        for i = 1:numel(gridData)
            if isfield(gridData(i), 'GreenSpotData')
                spots = gridData(i).GreenSpotData;
                if isstruct(spots)
                    for j = 1:length(spots)
                        if isfield(spots(j), 'colocFarRed') && spots(j).colocFarRed == 1
                            if isfield(spots(j), 'intensityTrace')
                                allTraces(end+1).trace = spots(j).intensityTrace; %#ok<AGROW>
                                allTraces(end).stepCount = NaN; %#ok<AGROW>
                                allTraces(end).sourceFile = files{f};
                                allTraces(end).traceIndexInFile = j; %#ok<AGROW>
                            end
                        end
                    end
                end
            end
        end
    end

    if isempty(allTraces)
        errordlg('No colocalized Green traces found in selected files');
        return;
    end

    % Initialize UI
    fig = uifigure('Name', 'Colocalized Green Trace Viewer', 'Position', [100 100 700 400]);
    ax = uiaxes(fig, 'Position', [50 80 600 300]);
    nextBtn = uibutton(fig, 'push', 'Text', 'Next (n)', 'Position', [400 20 100 40], 'ButtonPushedFcn', @nextTrace);
    prevBtn = uibutton(fig, 'push', 'Text', 'Previous (p)', 'Position', [200 20 100 40], 'ButtonPushedFcn', @prevTrace);
    clearBtn = uibutton(fig, 'push', 'Text', 'Clear Step (c)', 'Position', [50 20 120 40], 'ButtonPushedFcn', @clearStep);
    exportBtn = uibutton(fig, 'push', 'Text', 'Export', 'Position', [550 20 100 40], 'ButtonPushedFcn', @exportData);
    saveTrainingBtn = uibutton(fig, 'push', 'Text', 'Save Training Data', 'Position', [550 70 100 40], 'ButtonPushedFcn', @saveTrainingData);
    infoLbl = uilabel(fig, 'Text', '', 'Position', [180 20 150 40]);

    % State
    currentIndex = 1;
    randomOrder = randperm(length(allTraces));
    allTraces = allTraces(randomOrder);
    numTraces = length(allTraces);
    updatePlot();

    % Keypress support
    fig.KeyPressFcn = @(src, event) keyCallback(event);

    function updatePlot()
        cla(ax);
        trace = allTraces(currentIndex).trace;
        traceMean = mean(trace);
        traceStd = std(trace);
        if traceStd == 0
            traceStd = 1; % avoid division by zero
        end
        trace = (trace - traceMean) / traceStd;
        plot(ax, trace);
        title(ax, sprintf('Trace %d of %d', currentIndex, numTraces));
        xlabel(ax, 'Frame');
        ylabel(ax, 'Fluorescence Intensity');
        infoLbl.Text = sprintf('Trace #%d', currentIndex);

        % Display assigned step count in top right
        if ~isnan(allTraces(currentIndex).stepCount)
            text(ax, length(trace) * 0.95, max(trace) * 0.95, ...
                sprintf('%d step(s)', allTraces(currentIndex).stepCount), ...
                'HorizontalAlignment', 'right', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end

    function nextTrace(~, ~)
        currentIndex = currentIndex + 1;
        if currentIndex > numTraces
            currentIndex = 1;
        end
        updatePlot();
    end

    function prevTrace(~, ~)
        currentIndex = currentIndex - 1;
        if currentIndex < 1
            currentIndex = numTraces;
        end
        updatePlot();
    end

    function clearStep(~, ~)
        allTraces(currentIndex).stepCount = NaN;
        updatePlot();
    end

    function keyCallback(event)
        switch event.Key
            case {'numpad0','0'}
                assignStep(0);
            case {'numpad1','1'}
                assignStep(1);
            case {'numpad2','2'}
                assignStep(2);
            case {'numpad3','3'}
                assignStep(3);
            case {'numpad4','4'}
                assignStep(4);
            case {'numpad5','5'}
                assignStep(5);
            case {'numpad6','6'}
                assignStep(6);
            case {'numpad7','7'}
                assignStep(7);
            case {'numpad8','8'}
                assignStep(8);
            case {'numpad9','9'}
                assignStep(9);
            case 'c'
                clearStep();
            case 'n'
                nextTrace();
            case 'p'
                prevTrace();
        end
    end

    function assignStep(val)
        allTraces(currentIndex).stepCount = val;

        % Auto-save every 1000 traces tagged
        numTagged = sum(~isnan([allTraces.stepCount]));
        if mod(numTagged, 1000) == 0
            timestamp = datestr(now, 'yyyymmdd_HHMMSS');
            autosaveFile = fullfile(tempdir, ['autosave_stepCounts_' timestamp '.mat']);
            save(autosaveFile, 'allTraces');
            disp(['Auto-saved at ' autosaveFile]);
        end
        updatePlot();
    end

    function exportData(~, ~)
        % Create histogram
        counts = [allTraces.stepCount];
        stepHist = histcounts(counts(~isnan(counts)), 0:20); % Up to 19 steps
        numRejected = sum(isnan(counts));

        [file, path] = uiputfile({'*.xlsx'}, 'Save Step Count Data as Excel');
        if isequal(file,0)
            return;
        end

        % Create table and write to Excel
        traceIndices = num2cell(1:length(allTraces))';
        stepCounts = num2cell([allTraces.stepCount]');
        sourceFiles = {allTraces.sourceFile}';
        traceInFile = num2cell([allTraces.traceIndexInFile]');
        stepData = [traceIndices, stepCounts, sourceFiles, traceInFile];
        T = cell2table(stepData, 'VariableNames', {'TraceIndex','StepCount','SourceFile','TraceIndexInFile'});
        histogramTable = table((0:19)', stepHist', 'VariableNames', {'StepCount','Frequency'});
        rejectedTable = table(numRejected, 'VariableNames', {'NumRejected'});

        writetable(T, fullfile(path, file), 'Sheet', 'TraceData');
        % Create per-file histograms
        uniqueFiles = unique(sourceFiles);
        for i = 1:length(uniqueFiles)
            fileMask = strcmp(sourceFiles, uniqueFiles{i});
            countsPerFile = [allTraces(fileMask).stepCount];
            stepHistPerFile = histcounts(countsPerFile(~isnan(countsPerFile)), 0:20);
            fileHistTable = table((0:19)', stepHistPerFile', 'VariableNames', {'StepCount','Frequency'});
            [~, sheetName] = fileparts(uniqueFiles{i});
            sheetName = matlab.lang.makeValidName(sheetName);
            writetable(fileHistTable, fullfile(path, file), 'Sheet', sheetName);
        end

        writetable(histogramTable, fullfile(path, file), 'Sheet', 'Histogram');
        writetable(rejectedTable, fullfile(path, file), 'Sheet', 'Rejected');

        uialert(fig, 'Step count data exported to Excel.', 'Export Complete');
    end

    function saveTrainingData(~, ~)
        [file, path] = uiputfile({'*.mat'}, 'Save Training Data');
        if isequal(file, 0)
            return;
        end
        numTraces = length(allTraces);
        traceLength = length(allTraces(1).trace);
        
        X = zeros(numTraces, traceLength);        % Normalized traces
        X_raw = zeros(numTraces, traceLength);    % Raw traces
        y = [allTraces.stepCount]';
        sourceFile = {allTraces.sourceFile}';
        traceIndexInFile = [allTraces.traceIndexInFile]';
        
        for i = 1:numTraces
            raw = allTraces(i).trace;
            traceMean = mean(raw);
            traceStd = std(raw);
            if traceStd == 0
                traceStd = 1;
            end
            norm = (raw - traceMean) / traceStd;
        
            X(i, :) = norm;
            X_raw(i, :) = raw;
        end
        traces = X;     % Renamed for Python compatibility
        labels = y;     % Renamed for Python compatibility
        
        save(fullfile(path, file), 'traces', 'labels', 'X_raw', 'sourceFile', 'traceIndexInFile');
        uialert(fig, 'Training data (raw + normalized) saved as .mat file.', 'Save Complete');
    end
end
