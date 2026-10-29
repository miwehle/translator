# AGENTS.md

## Gemeinsame Basis

Lies vor der Arbeit [AGENTS_BASE.md](../../../dev/agents-base/AGENTS_BASE.md). Die dortigen Regeln einschließlich der eingebundenen Testregeln gelten verbindlich für dieses Repository. Die folgenden Regeln ergänzen diese Basis.

## Workspace-Struktur

Dies ist ein Multi-Ordner-Workspace.

Zu diesem Workspace gehoeren die Ordner `translator`, `data_preprocessor` und `model_based_curation`.

Wenn in Anfragen `data_preprocessor` erwaehnt wird, ist damit in diesem Workspace der eigenstaendige Workspace-Ordner `data_preprocessor` gemeint oder das darin enthaltene Hauptpaket gleichen Namens, nicht ein Unterordner von `translator`.
Analog für `model_based_curation` und `lab_infrastructure`.

## Refactorings

Keine stillschweigende Code-Duplizierung bei Refactorings; Duplikationen muessen ausdruecklich benannt werden.

Provisorische Workarounds, Debug-Helfer und asymmetrische Zwischenloesungen sind nach Klaerung der Ursache im selben Task wieder zu entfernen oder explizit zu begruenden; kein liegengebliebenes "temporary fix".

## Production-Code-Aenderungen

Aenderungen im Paket `model` nur nach Ruecksprache.

Im Paket `model` auf keinen Fall voreilig oder auf Verdacht aendern.

## Kleinste wirksame Aenderung

Erst die kleinste wirksame Aenderung suchen.

## Kompakte Python-Schreibweise

Python-Code standardmäßig kompakt schreiben. Innerhalb der konfigurierten maximalen Zeilenlänge ist im Zweifel die kompaktere Form zu bevorzugen.

Funktionsköpfe und besonders Funktionsaufrufe nicht vorschnell vertikal aufbrechen. Ein-Parameter-pro-Zeile-Layouts sind nicht der Default. Sie sind nur sinnvoll, wenn die kompakte Form die Zeilenlänge überschreitet oder fachlich klar schlechter lesbar wäre.

Mehrzeilige Python-Aufrufe sind möglichst kompakt zu schreiben. Zeilen, die nur aus einer schließenden Klammer bestehen, sind im Regelfall zu vermeiden. Funktionsaufrufe und ähnliche Konstrukte sollen erst bis zur maximal erlaubten Zeilenlänge horizontal wachsen, bevor sie vertikal wachsen.

Ein häufiges unnötiges Muster ist ein abschließendes Komma im letzten Argument, gefolgt von einer eigenen Zeile nur mit der schließenden Klammer. Wenn das Entfernen dieses letzten Kommas eine kompaktere und weiterhin klar lesbare Form innerhalb der maximal erlaubten Zeilenlänge ermöglicht, ist diese Form zu bevorzugen.

Zur Einordnung: In einer Stichprobe des Hugging Face Transformers-Codes liegt der Anteil von Zeilen, die nur aus einer schließenden Klammer bestehen, grob bei 3 %. Ein deutlich höherer Anteil ist in diesem Workspace nicht erwünscht.

Bestehende kompakte Aufruf-Layouts nicht ohne fachlichen Grund aufspreizen. Reine Stil-Umbauten hin zu mehr vertikaler Länge vermeiden.

Positionale Parameter bevorzugen, wenn der Aufruf dadurch kürzer und trotzdem klar bleibt. Keyword-only-Parameter nur mit klarem Mehrwert für Lesbarkeit, Sicherheit oder Eindeutigkeit. Aufgeblähte Funktionsaufrufe durch unnötige Keywords vermeiden.

Bei kleinen oder lokalen Änderungen sind kleiner Diff und geringe LOC wichtiger als stilistische Umformungen ohne fachlichen Nutzen.

Keine Einzeiler für Funktionsdefinitionen. Zwischen Funktionssignatur und Funktionsrumpf steht immer ein Zeilenumbruch.

Die im jeweiligen Repo konfigurierte Tooling-Konfiguration ist zu beachten, insbesondere `ruff` in `pyproject.toml` inklusive `line-length` und Stilregeln.

## Lokale Hilfsfunktionen

Lokale Hilfsfunktionen innerhalb einer Funktion oder Methode nicht mitten im Hauptfluss definieren. Wenn eine lokale Hilfsfunktion sinnvoll ist, dann am Anfang des umschließenden Blocks platzieren oder als eigene private Funktion/Methode auslagern. Der Scope ist dabei so klein wie möglich zu halten: Eine Hilfsfunktion soll nur dort sichtbar sein, wo sie fachlich benötigt wird, aber den Hauptfluss nicht unterbrechen. Ziel ist, dass der Hauptfluss ohne Unterbrechung lesbar bleibt und Hilfslogik bei Bedarf separat nachgeschlagen werden kann.

## Temp-Artefakte

Temporäre Verzeichnisse und Dateien für Tests, Verifikation und ad-hoc Läufe sind im Repo-Root ausschließlich unter `.local_tmp/` anzulegen.

Keine neuen temporären Root-Ordner wie `.tmp_pytest*`, `.pytest_tmp*` oder ähnliche Namen anlegen.

Pytest normal starten. Nur bei konkreten Sandbox-/Permission-Fehlern mit begruendeter Eskalation erneut ausfuehren.
