import io
import json
import os
import subprocess
import time

from IPython.core.display_functions import clear_output
from google.oauth2 import service_account
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaIoBaseDownload

from SRC.CORE._CONFIGS import SERVICE_ACCOUNT_FILE
from SRC.CORE._CONSTANTS import project_root_dir
from SRC.LIBRARIES.email_utils import send_email__DATA_drive_file_id__updated_alert


FOLDER_ID = '1hRWnY8Ae7DFnyi3G2SP5d1C_RJIH75rB'
GOOGEL_DRIVE_OAUTH2_API_TOKEN_FILE = f'{project_root_dir()}/google_drive_token.json'
SCOPES = ['https://www.googleapis.com/auth/drive', 'https://www.googleapis.com/auth/drive.file']
FOLDER_NAME = 'DATA'


def AUTHORIZE_SAVE_TOKEN_GOOGLE_DRIVE_OAUTH2():
    flow = InstalledAppFlow.from_client_secrets_file('client_secret_899492721597-k220sq9r6lrj59q6vohmamrtlch6a7nm.apps.googleusercontent.com.json', SCOPES)

    creds = flow.run_local_server(port=0)

    token_data = {
        'token': creds.token,
        'refresh_token': creds.refresh_token,
        'token_uri': creds.token_uri,
        'client_id': creds.client_id,
        'client_secret': creds.client_secret,
        'scopes': creds.scopes
    }

    with open(GOOGEL_DRIVE_OAUTH2_API_TOKEN_FILE, 'w') as f:
        json.dump(token_data, f)

    print(f"✅ Saved {GOOGEL_DRIVE_OAUTH2_API_TOKEN_FILE}")


def PRODUCE_DRIVE_SERVICE():
    creds = Credentials.from_authorized_user_file(GOOGEL_DRIVE_OAUTH2_API_TOKEN_FILE, scopes=SCOPES)
    service = build('drive', 'v3', credentials=creds)

    return service


def list_files_in_folder(folder_id):
    service = PRODUCE_DRIVE_SERVICE()

    query = f"'{folder_id}' in parents and trashed = false"
    print(f"query: {query}")
    results = service.files().list(
        q=query,
        spaces='drive',
        fields="files(id, name, mimeType)"
    ).execute()

    files = results.get('files', [])
    print(f"\n📄 Files in folder {folder_id}:")
    if len(files) == 0:
        print(f"!There are NO files found!")
    else:
        for file in files:
            print(f"- {file['name']} (ID: {file['id']})")

    return files


def download_file_from_folder(file_id, file_name):
    uploadable_file_path = f'{project_root_dir()}/{file_name}'

    service = PRODUCE_DRIVE_SERVICE()
    request = service.files().get_media(fileId=file_id)
    fh = io.FileIO(file_name, 'wb')

    downloader = MediaIoBaseDownload(fh, request, chunksize=1024 * 1024 * 10)  # 10 MB chunks

    done = False
    last_progress = 0
    print(f"Download progress: {last_progress}%")
    while not done:
        status, done = downloader.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            if last_progress < progress:
                print(f"Download progress: {progress}%")
            last_progress = progress

    print(f"✅ File downloaded: {file_name}")


def upload_file_to_folder(file_name, folder_id=FOLDER_ID):
    uploadable_file_path = f'{project_root_dir()}/{file_name}'

    service = PRODUCE_DRIVE_SERVICE()

    file_metadata = {'name': file_name, 'parents': [folder_id]}
    media = MediaFileUpload(file_name, resumable=True)

    request = service.files().create(body=file_metadata, media_body=media, fields='id')
    response = None
    last_progress = 0
    print(f"Upload progress: {last_progress}%")
    while response is None:
        status, response = request.next_chunk()
        if status:
            progress = int(status.progress() * 100)
            if last_progress < progress:
                print(f"Upload progress: {progress}%")
            last_progress = progress

    uploaded_file_id = response.get('id')

    print(f"✅ Uploaded: {uploadable_file_path} (ID: {uploaded_file_id})")

    return uploaded_file_id


def delete_old_DATA_archives(folder_id: str):
    service = PRODUCE_DRIVE_SERVICE()

    query = (
        f"'{folder_id}' in parents and "
        f"trashed = false and "
        f"name contains 'DATA.tar.gz'"
    )

    results = service.files().list(
        q=query,
        spaces='drive',
        fields="files(id, name, createdTime)",
        orderBy="createdTime desc"  # newest first
    ).execute()

    files = results.get('files', [])

    if not files:
        print("⚠️ No DATA.tar.gz files found.")
        return

    # 2. Keep the newest one (first item)
    newest_file = files[0]
    old_files = files[1:]
    files_to_delete = files

    # 3. Delete all older ones
    for f in files_to_delete:
        try:
            service.files().delete(fileId=f['id']).execute()
            print(f"🗑️ Deleted old archive: {f['name']} (created {f['createdTime']})")
        except Exception as e:
            print(f"⚠️ Failed to delete {f['name']}: {e}")

    # print("✅ Cleanup complete — only latest DATA.tar.gz remains.")


def UPLOAD__DATA__FOLDER():
    from SRC.LIBRARIES.new_utils import run_safety_interrupter

    subprocess.run(['ls /workspace/CRYPTO_BOT/DATA/'], shell=True)
    subprocess.run(['ls /home/crypto/CRYPTO_BOT/DATA/'], shell=True)

    time.sleep(3)

    files = list_files_in_folder(FOLDER_ID)

    run_safety_interrupter("YOU SURE UPLOAD DATA FOLDER", count_down_secs=30)

    root_fodler_path = str(project_root_dir())

    folder_name = 'DATA'
    data_file_name = f"{folder_name}.tar.gz"
    subprocess.run([f"cd {project_root_dir()} && tar --use-compress-program=pigz -cvf {data_file_name} {folder_name}"], shell=True)

    clear_output(wait=False)

    print(f"COMPRESSED: {data_file_name}")

    delete_old_DATA_archives(FOLDER_ID)

    time.sleep(10)

    # FILE_ID = upload_file(data_file_name)
    FILE_ID = upload_file_to_folder(data_file_name)

    time.sleep(10)

    clear_output(wait=False)

    print(f"COMPRESSED: {data_file_name}")

    remove_archive_command = f"cd {root_fodler_path} && rm -rf {folder_name}.tar.gz"
    subprocess.run([remove_archive_command], shell=True)
    print(f"DELETE || File: {folder_name}.tar.gz")

    try:
        send_email__DATA_drive_file_id__updated_alert(f"{folder_name}.tar.gz", FILE_ID)

        print(f"COMPRESSED > UPLOADED > OLD ARCHIVE OVERRIDEN > DELETE > EMAIL SENT: {folder_name}.tar.gz")
    except Exception:
        print(f"EMAIL NOT SENT!")
        print(f"COMPRESSED > UPLOADED > OLD ARCHIVE OVERRIDEN > DELETE: {folder_name}.tar.gz")

    time.sleep(10)

    files = list_files_in_folder(FOLDER_ID)


def DOWNLOAD__DATA__FOLDER_LAST(remove_data_folder=False):
    from SRC.LIBRARIES.new_utils import run_safety_interrupter
    files = list_files_in_folder(FOLDER_ID)

    file_id = files[0]['id']

    root_fodler_path = str(project_root_dir())

    subprocess.run([f'ls {root_fodler_path}/CRYPTO_BOT/DATA/'], shell=True)

    time.sleep(3)

    destination_file = f'{root_fodler_path}/{FOLDER_NAME}.tar.gz'
    uncompress_archive_command = f"cd {root_fodler_path} && tar -xvf {FOLDER_NAME}.tar.gz"

    if remove_data_folder:
        run_safety_interrupter("YOU SURE REMOVE > DOWNLOAD DATA FOLDER", count_down_secs=30)

        subprocess.run([f'rm -rf {root_fodler_path}/DATA/'], shell=True)

    # download_file(file_id, destination_file)
    download_file_from_folder(file_id, destination_file)

    subprocess.run([uncompress_archive_command], shell=True)

    clear_output(wait=False)

    print(f"UNCOMPRESS ARCHIVE COMMAND: {uncompress_archive_command}")
    print(f"UNCOMPRESSED || File: {FOLDER_NAME}.tar.gz")

    remove_archive_command = f"cd {root_fodler_path} && rm -rf {FOLDER_NAME}.tar.gz"
    subprocess.run([remove_archive_command], shell=True)

    print(f"DELETE ARCHIVE COMMAND: {remove_archive_command}")
    print(f"DELETED: {FOLDER_NAME}.tar.gz")

    print(f"DOWNLOADED > UNCOMPRESS > DELETE: {FOLDER_NAME}.tar.gz")


if __name__ == "__main__":
    # files = list_files_in_folder(FOLDER_ID)
    DOWNLOAD__DATA__FOLDER_LAST()