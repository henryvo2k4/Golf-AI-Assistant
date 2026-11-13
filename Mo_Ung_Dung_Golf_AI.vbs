Set fso = CreateObject("Scripting.FileSystemObject")
Set WshShell = CreateObject("WScript.Shell")

currentDir = fso.GetParentFolderName(WScript.ScriptFullName)
pythonwExe = currentDir & "\.venv\Scripts\pythonw.exe"
appScript = currentDir & "\run_app.py"

If fso.FileExists(pythonwExe) Then
    WshShell.CurrentDirectory = currentDir
    WshShell.Run """" & pythonwExe & """ """ & appScript & """", 0, False
Else
    MsgBox "Khong tim thay moi truong Python tai: " & pythonwExe, 16, "Loi khoi dong VAR Golf"
End If
