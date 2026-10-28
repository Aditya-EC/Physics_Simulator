; Inno Setup script for the PSim Studio Windows installer.
;
; Build the frozen application first (on Windows):   python packaging\build_dist.py
; build_dist.py then calls:                            ISCC /DAppVersion=<version> packaging\windows\psim_studio.iss
;
; Built by CI on Windows with Inno Setup 6. With a code-signing certificate configured (packaging/signing.py),
; build_dist.py also passes /DSignInstaller and /Spsimsign=<signtool command>, and Inno Setup signs the setup
; and the uninstaller it embeds.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif
#define AppName "PSim Studio"
#define SrcDir "..\dist\PSimStudio"

[Setup]
AppId={{6B0F3E1C-5A0E-4D59-9E3B-2C1F7A9D4E51}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=PSim Studio
DefaultDirName={autopf}\PSim Studio
DefaultGroupName=PSim Studio
OutputDir=..\dist
OutputBaseFilename=PSimStudio-{#AppVersion}-setup
SetupIconFile=..\build\psim.ico
UninstallDisplayIcon={app}\psim-studio.exe
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequiredOverridesAllowed=dialog
ChangesAssociations=yes
WizardStyle=modern
#ifdef SignInstaller
SignTool=psimsign
SignedUninstaller=yes
#endif

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "associate"; Description: "Open .psim project files with PSim Studio"; GroupDescription: "File associations:"

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\PSim Studio"; Filename: "{app}\psim-studio.exe"
Name: "{group}\Example projects"; Filename: "{app}\examples"
Name: "{group}\{cm:UninstallProgram,PSim Studio}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\PSim Studio"; Filename: "{app}\psim-studio.exe"; Tasks: desktopicon

[Registry]
Root: HKA; Subkey: "Software\Classes\.psim"; ValueType: string; ValueName: ""; ValueData: "PSimStudio.Project"; Flags: uninsdeletevalue; Tasks: associate
Root: HKA; Subkey: "Software\Classes\PSimStudio.Project"; ValueType: string; ValueName: ""; ValueData: "PSim Studio project"; Flags: uninsdeletekey; Tasks: associate
Root: HKA; Subkey: "Software\Classes\PSimStudio.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\psim-studio.exe,0"; Tasks: associate
Root: HKA; Subkey: "Software\Classes\PSimStudio.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\psim-studio.exe"" ""%1"""; Tasks: associate

[Run]
Filename: "{app}\psim-studio.exe"; Description: "{cm:LaunchProgram,PSim Studio}"; Flags: nowait postinstall skipifsilent
