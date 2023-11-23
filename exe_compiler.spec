# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['exe_compiler.py'],
    pathex=[],
    binaries=[],
    datas=[('app.py','.'),('modules/*', 'modules'), ('plotting_tools/*', 'plotting_tools'), ('C:/Users/Endre/AppData/Local/Programs/Python/Python310/Lib/site-packages/shiny','shiny')],
    hiddenimports=['seaborn', 'scipy', 'shinyswatch'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='exe_compiler',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
