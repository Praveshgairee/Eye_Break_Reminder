# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=collect_dynamic_libs('mediapipe'),
    datas=[('models/blaze_face_short_range.tflite', 'models')],
    hiddenimports=collect_submodules('mediapipe') + collect_submodules('mediapipe.tasks'),
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Eye Break Reminder',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Eye Break Reminder',
)
app = BUNDLE(
    coll,
    name='Eye Break Reminder.app',
    icon=None,
    bundle_identifier='com.praveshgaire.eyebreakreminder',
    info_plist={
        'CFBundleName': 'Eye Break Reminder',
        'CFBundleDisplayName': 'Eye Break Reminder',
        'CFBundleIdentifier': 'com.praveshgaire.eyebreakreminder',
        'CFBundleShortVersionString': '1.0.0',
        'CFBundleVersion': '1.0.0',
        'LSUIElement': False,
        'NSCameraUsageDescription': 'This app uses the camera to detect when you are in front of the screen and remind you to rest your eyes.',
    },
)
