Get-ChildItem -Recurse -File |
Where-Object { $_.Name -match "\s+\.jpg$" -or $_.Name -match "\s+\.JPG$" } |
ForEach-Object {
    $newName = $_.Name -replace "\s+(?=\.[jJ][pP][gG]$)", ""
    Write-Host "Renaming '$($_.Name)' → '$newName'"
    Rename-Item -LiteralPath $_.FullName -NewName $newName -Force
}
