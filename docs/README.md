# CSI-VAD Project Page

Static project page for:

`Context-structured Video Anomaly Detection with Large Vision-Language Models`

The page is based on the Academic Project Page Template:

https://github.com/eliahuhorwitz/Academic-project-page-template

## Structure

- `index.html`: CSI-VAD project page.
- `.nojekyll`: disables Jekyll processing on GitHub Pages.
- `static/css/`: template CSS plus `csi-vad.css` overrides.
- `static/js/`: template JavaScript assets.
- `static/images/`: framework, recognition module, social preview, and video posters.
- `static/videos/`: MP4 files for qualitative examples.

## Deploy to GitHub Pages

Copy this directory into the `docs/` folder of `dj991108/CSI-VAD`, then push:

```bash
rsync -av --delete \
  /home1/irteam/data-vol1/Surveillance_VAD/docs/csi_vad/project_page/ \
  /tmp/CSI-VAD/docs/

cd /tmp/CSI-VAD
git add docs
git commit -m "Use academic project page template"
git push origin main
```

In GitHub, set Pages to:

- Source: `Deploy from a branch`
- Branch: `main`
- Folder: `/docs`

Expected URL:

https://dj991108.github.io/CSI-VAD/

## Remaining Updates

- Replace `arXiv coming soon` when the arXiv URL is available.
- Replace `Code coming soon` when code release is ready.
- Update the camera-ready LaTeX URL after this page is public.
