# YOVO-3M and YOVO-10M datasets
We provide the newly-created large-scale web video datasets YOVO-3M and YOVO-10M with the format of original video id and related meta data (e.g., searched query and video title). 

Since all web videos are collected from YouTube, you can obtain the video URL through the combination of https://www.youtube.com/watch?v= + video_id. (e.g., https://www.youtube.com/watch?v=pReZC78Hb-s) 

## Download Link
[Baidu Cloud]
https://pan.baidu.com/s/17wR4uYNqMfjLUo0KhXJBuQ (code: dzel) 

## YOVO-3M Formats
In the zip file of `YOVO-3M.zip`, there are four files in it:
- YOVO-3M_video.csv
- YOVO-3M_query.txt
- YOVO-3M_video_id.txt
- YOVO-3M_video_title.txt

### YOVO-3M_video.csv
Each line in this file contains a video id, the start time and end time (second) for the sampled video clip and the corresponding query id.

### YOVO-3M_query.txt
Each line in this file contains a query label. The query id in `YOVO-3M_video.csv` file is the column index of this file.

### YOVO-3M_video_id.txt
Each line in this file contains a video id.

### YOVO-3M_video_title.txt
Each line in this file contains a title for each video, the corresponding video id is in the same line of `YOVO-3M_video_id.txt`.

## YOVO-10M Formats
The YOVO-10M is the extension of YOVO-3M. The extended query vocabulary is generated from the Oxford English Dictionary which includes two parts, i.e., the verb vocabulary (v_word) and the noun vocabulary (n_word). 

We provide the extended original video id and related meta data in the `YOVO-10M.zip`. 

In the zip file of YOVO-10M, there are eight files in it:
- n(v)_word_video.csv
- n(v)_word_query.txt
- n(v)_word_video_id.txt
- n(v)_word_video_title.txt

### n(v)_word_video.csv
Each line in these two files contains a video id, the start time and end time (second) for the sampled video clip and the corresponding query id.

### n(v)_word_query.txt
Each line in these two files contains a query label. The query id in `n(v)_word_video.csv` file is the column index of the `n(v)_word_query.txt` file.

### n(v)_word_video_id.txt
Each line in these two files contains a video id.

### n(v)_word_video_title.txt
Each line in these two file contains a title for each video, the corresponding video id is in the same line of `n(v)_word_video_id.txt`.
