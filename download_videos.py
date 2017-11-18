# -*- coding: utf-8 -*-
"""
Download list of videos from youtube
"""
import csv
import youtube_dl as yt
import argparse
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("-url_csv", required=True, help="CSV file where the url of the videos to download are.")
parser.add_argument("-full_videos_path", required=True, help="Output folder of the dataset.")
parser.add_argument("-cropped_videos_path", required=True, help="Output folder of the dataset cropped videos.")
args = parser.parse_args()


def read_urls(file_read):

    urls_r = []
    with open(file_read) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            urls_r.append({"url": row["url"], "interval": row["interval"]})
    return urls_r


def download(url_list):
    error_counter = 0
    count = 0

    for url in url_list:
        count += 1
        print "Downloading video {}/{} with url [{}]".format(count, len(url_list), url['url'])
        item = url['url'].split('=')[-1] + '.mp4'
        item_path = unicode(os.path.join(args.full_videos_path, item))
        try:
            options = {
                'outtmpl': item_path,
                'format': "22"
            }
            with yt.YoutubeDL(options) as ydl:
                ydl.download([url['url']])
        except Exception:
            print "Download error."
            error_counter += 1
        # ffmpeg -i input_file -ss 00:00:15.00 -t 00:00:10.00 -c copy out.mp4
        item_cropped_path = os.path.join(args.cropped_videos_path, item)
        start_interval = url["interval"].split("-")[0]
        finish_interval = url["interval"].split("-")[1]
        print 'ffmpeg -i ' + item_path + ' -ss ' + start_interval + ' -to ' + finish_interval + ' -c ' + item_cropped_path
        os.system('ffmpeg -i ' + item_path + ' -ss ' + start_interval + ' -to ' + finish_interval + ' -c copy '
                  + item_cropped_path)

    print "Found {} errors".format(error_counter)


urls = read_urls(args.url_csv)
download(urls)
print "Finished download"




