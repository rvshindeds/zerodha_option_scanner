Set WshShell = CreateObject("WScript.Shell")

' Get the folder where this .vbs file lives
scriptPath = WScript.ScriptFullName
folderPath = Left(scriptPath, InStrRev(scriptPath, "\"))

' Build path to the .bat in the same folder
batPath = folderPath & "start_scanner.bat"

' Run hidden (0)
WshShell.Run Chr(34) & batPath & Chr(34), 0

Set WshShell = Nothing
