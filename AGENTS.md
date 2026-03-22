## Workspace-Struktur

Dies ist ein Multi-Ordner-Workspace.

Zu diesem Workspace gehoeren die Ordner `translator` und `data_preprocessor`.

Wenn in Anfragen `data_preprocessor` erwaehnt wird, ist damit in diesem Workspace der eigenstaendige Workspace-Ordner `data_preprocessor` gemeint oder das darin enthaltene Hauptpaket gleichen Namens, nicht ein Unterordner von `translator`.

## Refactorings

Keine stillschweigende Code-Duplizierung bei Refactorings; Duplikationen muessen ausdruecklich benannt werden.

Provisorische Workarounds, Debug-Helfer und asymmetrische Zwischenloesungen sind nach Klaerung der Ursache im selben Task wieder zu entfernen oder explizit zu begruenden; kein liegengebliebenes "temporary fix".

## Temp-Artefakte

Temporäre Verzeichnisse und Dateien für Tests, Verifikation und ad-hoc Läufe sind im Repo-Root ausschließlich unter `.local_tmp/` anzulegen.

Keine neuen temporären Root-Ordner wie `.tmp_pytest*`, `.pytest_tmp*` oder ähnliche Namen anlegen.
