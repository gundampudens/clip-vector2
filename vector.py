import argparse, json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from tqdm import tqdm

def main(sa_path, folder_id, output):
    creds = service_account.Credentials.from_service_account_file(
        sa_path, scopes=['https://www.googleapis.com/auth/drive.readonly'])
    service = build('drive', 'v3', credentials=creds)

    query = f"'{folder_id}' in parents and trashed=false and mimeType contains 'image/'"
    files, page_token = [], None
    pbar = tqdm(unit="file")
    while True:
        resp = service.files().list(
            q=query, spaces='drive',
            fields='nextPageToken, files(id,name)',
            pageSize=1000, pageToken=page_token
        ).execute()
        for f in resp.get('files', []):
            url = f"https://drive.google.com/uc?id={f['id']}"
            files.append({'name': f['name'], 'url': url})
            pbar.update(1)
        page_token = resp.get('nextPageToken')
        if not page_token: break
    pbar.close()

    with open(output, 'w', encoding='utf-8') as fp:
        json.dump(files, fp, ensure_ascii=False, indent=2)
    print(f"Saved {len(files)} entries to {output}")

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--sa', required=True)
    p.add_argument('--folder', required=True)
    p.add_argument('--output', default='drive_images.json')
    args = p.parse_args()
    main(args.sa, args.folder, args.output)
