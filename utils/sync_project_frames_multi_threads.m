% The directory where you extracted the raw dataset.
datasetDir = '../../../dataset/nyu_depth_v2/raw';
% The directory where you want save the synced images and depths.
dstDir = '../../../dataset/nyu_depth_v2/sync';

fid = fopen('train_scenes.txt', 'r');
tline = fgetl(fid);
sceneNames = cell(0, 1);
while ischar(tline)
    sceneNames{end+1, 1} = tline;
    tline = fgetl(fid);
end
fclose(fid);

num_threads = 6;
sample_step = 7;

parpool(num_threads);

for aa = 1 : num_threads : numel(sceneNames)
    actual_num_threads = min(numel(sceneNames) - aa + 1, num_threads);
    sceneNames_batch = sceneNames(aa:aa+actual_num_threads-1);
    parfor i = 1:actual_num_threads
        sceneName = sceneNames_batch{i};
        % The absolute directory of the
        sceneDir = sprintf('%s/%s', datasetDir, sceneName);
        
        % Reads the list of frames.
        frameList = get_synched_frames(sceneDir);
        saveDir = sprintf('%s/%s', dstDir, sceneName)
        if ~exist(saveDir, 'dir')
            % Folder does not exist so create it.
            mkdir(saveDir);
        end

        ind = 0;
        
        % Displays each pair of synchronized RGB and Depth frames.
        for ii = 1 : sample_step : numel(frameList)
            imgRgb = imread([sceneDir '/' frameList(ii).rawRgbFilename]);
            if frameList(ii).rawDepthFilename == "d-1315166703.129542-2466101449.pgm" % Faulty image
                continue;
            end
            imgDepthRaw = swapbytes(imread([sceneDir '/' frameList(ii).rawDepthFilename]));
            [imgDepthProj, imgRgbUd] = project_depth_map(imgDepthRaw, imgRgb);

            rgb_dst = sprintf('%s/rgb_%05d.jpg', saveDir, ind);
            imwrite(imgRgbUd, rgb_dst);

            imgDepthProj = uint16(imgDepthProj * 1000.0);
            sync_depth_dst = sprintf('%s/sync_depth_%05d.png', saveDir, ind);                        
            imwrite(imgDepthProj, sync_depth_dst);
            
            ind = ind + 1;
            fprintf('%d/%d done\n', ii, numel(frameList));
        end
        fprintf('%s done', sceneName);
    end
end