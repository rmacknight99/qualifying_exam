# Import Google libraries
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from pydrive.files import GoogleDriveFileList
import googleapiclient.errors

# Import general libraries
from argparse import ArgumentParser
from os import chdir, listdir, stat
import sys
from datetime import date


def copy_to_gdrive(src_folder_name, shared_ID=None):
    drive = GoogleDrive(GoogleAuth())
    today = date.today()
    folder_list = drive.ListFile({'q': f"'{shared_ID}' in parents and trashed=false"}).GetList()
    dst_folder_name = "campaign_"+today.strftime("%b_%d_%Y")
    create = True
    
    for folder in folder_list:
        if folder['title'] == dst_folder_name:
            create = False
            folder_id = folder['id']
    if create:
        folder_id = create_folder(drive, f"{shared_ID}", dst_folder_name = "campaign_"+today.strftime("%b_%d_%Y"))
    
    chdir(src_folder_name)
    print(f'\ncopying files to campaign_{today.strftime("%b_%d_%Y")} google drive')
    for content in listdir('.'):
        statinfo = stat(content)
        if statinfo.st_size > 0:
            try:
                f = drive.CreateFile({"parents": [{"kind": "drive#fileLink", "id": folder_id}]})
                f.SetContentFile(content)
                print('\tuploading ' + content)
                f.Upload()
            except:
                pass
        else:
            print('\tfile {0} is empty'.format(content))

def create_folder(drive, ID, dst_folder_name):
    print('creating folder ' + dst_folder_name)
    parent_folder_id = ID
    folder_metadata = {'title': dst_folder_name,
                       'mimeType': 'application/vnd.google-apps.folder',
                       'parents': [{"kind": "drive#fileLink", "id": parent_folder_id}]}
    folder = drive.CreateFile(folder_metadata)
    folder.Upload()
    print('\ttitle: %s, id: %s' % (folder['title'], folder['id']))
    folder_id = folder['id']
    return folder_id
