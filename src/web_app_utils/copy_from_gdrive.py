# Import Google libraries
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFileList
import googleapiclient.errors
from apiclient import discovery

# Import general libraries
from argparse import ArgumentParser
from os import chdir, listdir, stat
import sys

def copy_from_gdrive(search_name="UCONN_CMU", drive_ID=None):
    drive = GoogleDrive(GoogleAuth())
    try:
        file_list = drive.ListFile({'q': f"'{drive_ID}' in parents and trashed=false"}).GetList()
    except:
        print(f'path to {search_name} not found')

    download_file(drive, file_list)
    
def download_file(drive, file_list, search_name="UCONN_CMU", dst_folder_name="data_folder", dst_file_name="raw_data.csv"):
    print(f'copying files from google drive to {dst_folder_name}')
    mimetypes = {'application/vnd.google-apps.spreadsheet': 'text/csv'}
    for f1 in file_list:
        if f1['title'] == search_name:
            if f1['mimeType'] in mimetypes:
                download_mimetype = mimetypes[f1['mimeType']]
                content = f1.GetContentString(mimetype=download_mimetype)
            else:
                content = f1.GetContentString()
            print(f"found {dst_file_name}")
    with open(dst_folder_name+"/"+dst_file_name, "w") as new_file:
        new_file.write(content)
