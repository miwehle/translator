## Workspace-Struktur

Dies ist ein Multi-Ordner-Workspace.

Zu diesem Workspace gehoeren die Ordner `translator`, `data_preprocessor` und `model_based_curation`.

Wenn in Anfragen `data_preprocessor` erwaehnt wird, ist damit in diesem Workspace der eigenstaendige Workspace-Ordner `data_preprocessor` gemeint oder das darin enthaltene Hauptpaket gleichen Namens, nicht ein Unterordner von `translator`.
Analog für `model_based_curation`.

## Refactorings

Keine stillschweigende Code-Duplizierung bei Refactorings; Duplikationen muessen ausdruecklich benannt werden.

Provisorische Workarounds, Debug-Helfer und asymmetrische Zwischenloesungen sind nach Klaerung der Ursache im selben Task wieder zu entfernen oder explizit zu begruenden; kein liegengebliebenes "temporary fix".

## Struktur und Benennung von Testcode

### 1 Grundprinzip
1.1 Testcode soll klar strukturiert und klar benannt sein.  
1.2 Testcode folgt einer 1:1-Korrespondenz zum Production-Code.  
1.3 Abweichungen davon sind zu begründen.
1.4 Auch fuer Unit-Tests gilt das KISS-Prinzip. Testcode soll moeglichst klein, zielgerichtet und mit geringer mentaler Last bleiben.
1.5 Bestehenden Testcode bevorzugt geschickt wiederverwenden und erweitern, wenn dadurch Diff, `LOC` und Redundanz kleiner bleiben.
1.6 Auch bei Testcode aktiv prüfen, ob eine kleine Vereinfachung oder ein kleines Refactoring bestehenden Testcode vereinfachen und `LOC` sparen kann.
1.7 Vor groesseren Refactorings am Testcode kurz beschreiben, was vereinfacht werden soll, und ein Go einholen.

### 2 Definition von "öffentlich"
2.1 Öffentlich ist ausschließlich, was über das zuständige `__init__.py` öffentlich gemacht wird.  
2.2 Das betrifft Module, Klassen und Funktionen.  

### 3 Strukturierung von Testcode
3.1 Production-Modul <-> korrespondierendes Test-Modul.  
3.2 Production-Klasse <-> im Regelfall korrespondierende Test-Klasse.  
3.3 Production-Funktion oder -Methode <-> korrespondierende Test-Funktion oder Test-Methode.  
3.4 Diese Zuordnung ist verbindlich. Abweichungen sind zu begründen.
3.5 Tests priorisieren die öffentliche API. Sonst entstehen leicht Tests, die fragil sind oder Refactorings des Production-Codes erschweren.
3.6 Nicht-öffentliche Objekte können eigene Tests haben, wenn sie substanzielle Fachlogik tragen oder die Tests über die öffentliche API unhandlich würden. Auch solche Tests folgen möglichst der Struktur des Production-Codes.
3.7 Nicht jedes öffentliche Production-Objekt benötigt einen eigenen Test. Der Schwerpunkt soll dort liegen, wo die Dichte der Fachlogik hoch ist; triviale Getter sind im Regelfall kein eigener Testgegenstand.

### 4 Integrationstests
4.1 Tests für Zusammensetzungen mehrerer Production-Bausteine sind erlaubt.  
4.2 Sie gehören nicht zur 1:1-Zuordnung.  
4.3 Sie liegen getrennt unter `tests/integration`.

### 5 Benennung von Testcode
5.1 Testnamen sollen den getesteten Production-Code direkt erkennbar machen.  
5.2 Ein beschreibender Suffix am Ende des Namens ist erlaubt.  
5.3 Testdatei: `test_<modulname>.py`.  
5.4 Testklasse: `Test<Klassenname>`.  
5.5 Testmethode: `test_<methodenname>...`.  
5.6 Freie Testfunktion: `test_<funktionsname>...`.  
5.7 Der Modulname soll nicht in der Testfunktion wiederholt werden.  
5.8 Der Klassenname soll nicht in der Testmethode wiederholt werden, wenn die Testklasse ihn schon trägt.

### Nutzen und Zweck
- Die 1:1-Zuordnung schafft Ordnung im Testcode. Sie macht klar, wo welcher Test zu erwarten ist und was getestet ist und was nicht.
- Das Regelwerk lenkt Testaufwand auf die öffentliche API. So wächst der Testcode nicht ungeordnet entlang interner Implementierungsdetails.
- Das Regelwerk folgt bewährten Testmustern aus Gerard Meszaros, *xUnit Test Patterns*, insbesondere der klaren Zuordnung im Sinne von `Testcase Class per Class` und `Test Method`. Es verbessert so Auffindbarkeit, Verständlichkeit und Wartbarkeit des Testcodes.

## Production-Code-Aenderungen

Aenderungen im Paket `model` nur nach Ruecksprache.

Im Paket `model` auf keinen Fall voreilig oder auf Verdacht aendern.

Bei nicht-trivialen Aenderungen am Production-Code vor der Umsetzung kurz und buendig beschreiben, was geaendert werden soll, und ein Go einholen.

KISS-Prinzip beachten: Production-Code moeglichst einfach und klein halten; Zusatzlogik und Diagnose nur behalten, wenn ihr Nutzen die Komplexitaet klar rechtfertigt.

## Kleinste wirksame Aenderung

Erst die kleinste wirksame Aenderung suchen.

KISS-Prinzip beachten: `LOC` und mentale Last sind als harte Kosten zu behandeln. Zusaetzliche `LOC` sind nur gerechtfertigt, wenn sie einen klaren fachlichen Mehrwert bringen, vor allem in Form zusaetzlicher oder deutlich besserer Funktion.

## Kompakte Python-Schreibweise

Python-Code standardmäßig kompakt schreiben. Innerhalb der konfigurierten maximalen Zeilenlänge ist im Zweifel die kompaktere Form zu bevorzugen.

Funktionsköpfe und besonders Funktionsaufrufe nicht vorschnell vertikal aufbrechen. Ein-Parameter-pro-Zeile-Layouts sind nicht der Default. Sie sind nur sinnvoll, wenn die kompakte Form die Zeilenlänge überschreitet oder fachlich klar schlechter lesbar wäre.

Mehrzeilige Python-Aufrufe sind möglichst kompakt zu schreiben. Zeilen, die nur aus einer schließenden Klammer bestehen, sind im Regelfall zu vermeiden. Funktionsaufrufe und ähnliche Konstrukte sollen erst bis zur maximal erlaubten Zeilenlänge horizontal wachsen, bevor sie vertikal wachsen.

Ein häufiges unnötiges Muster ist ein abschließendes Komma im letzten Argument, gefolgt von einer eigenen Zeile nur mit der schließenden Klammer. Wenn das Entfernen dieses letzten Kommas eine kompaktere und weiterhin klar lesbare Form innerhalb der maximal erlaubten Zeilenlänge ermöglicht, ist diese Form zu bevorzugen.

Zur Einordnung: In einer Stichprobe des Hugging Face Transformers-Codes liegt der Anteil von Zeilen, die nur aus einer schließenden Klammer bestehen, grob bei 3 %. Ein deutlich höherer Anteil ist in diesem Workspace nicht erwünscht.

Bestehende kompakte Aufruf-Layouts nicht ohne fachlichen Grund aufspreizen. Reine Stil-Umbauten hin zu mehr vertikaler Länge vermeiden.

Positionale Parameter bevorzugen, wenn der Aufruf dadurch kürzer und trotzdem klar bleibt. Keyword-only-Parameter nur mit klarem Mehrwert für Lesbarkeit, Sicherheit oder Eindeutigkeit. Aufgeblähte Funktionsaufrufe durch unnötige Keywords vermeiden.

Zusätzliche Hilfsfunktionen, Basismodule und Abstraktionen nur einführen, wenn sie echte Wiederverwendung oder klare fachliche Vereinfachung bringen. Ein bloß generischerer oder vermeintlich saubererer Stil rechtfertigt keinen zusätzlichen Code.

Bei kleinen oder lokalen Änderungen sind kleiner Diff und geringe LOC wichtiger als stilistische Umformungen ohne fachlichen Nutzen.

Keine Einzeiler für Funktionsdefinitionen. Zwischen Funktionssignatur und Funktionsrumpf steht immer ein Zeilenumbruch.

Die im jeweiligen Repo konfigurierte Tooling-Konfiguration ist zu beachten, insbesondere `ruff` in `pyproject.toml` inklusive `line-length` und Stilregeln.

## Lokale Hilfsfunktionen

Lokale Hilfsfunktionen innerhalb einer Funktion oder Methode nicht mitten im Hauptfluss definieren. Wenn eine lokale Hilfsfunktion sinnvoll ist, dann am Anfang des umschließenden Blocks platzieren oder als eigene private Funktion/Methode auslagern. Der Scope ist dabei so klein wie möglich zu halten: Eine Hilfsfunktion soll nur dort sichtbar sein, wo sie fachlich benötigt wird, aber den Hauptfluss nicht unterbrechen. Ziel ist, dass der Hauptfluss ohne Unterbrechung lesbar bleibt und Hilfslogik bei Bedarf separat nachgeschlagen werden kann.

## Temp-Artefakte

Temporäre Verzeichnisse und Dateien für Tests, Verifikation und ad-hoc Läufe sind im Repo-Root ausschließlich unter `.local_tmp/` anzulegen.

Keine neuen temporären Root-Ordner wie `.tmp_pytest*`, `.pytest_tmp*` oder ähnliche Namen anlegen.
