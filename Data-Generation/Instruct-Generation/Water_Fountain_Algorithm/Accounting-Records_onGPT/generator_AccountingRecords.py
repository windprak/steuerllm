from io import BytesIO
import json
import random
import time
from typing import Dict, List
from PyPDF2 import PdfReader
from bs4 import BeautifulSoup
import numpy as np
from ollama import chat
from pydantic import BaseModel
import requests
from sklearn.metrics.pairwise import cosine_similarity
from openai import AzureOpenAI
from nltk.tokenize import sent_tokenize
from PyPDF2.errors import PdfReadError

# Azure API-Setup
AZURE_API_KEY = ""
AZURE_ENDPOINT = ""
AZURE_API_VERSION = ""
AZURE_DEPLOYMENT_ID = ""

searxng_url = "http://localhost:8080"

# Initialise Azure OpenAI client
llm = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version=AZURE_API_VERSION
)

# Starting point for the history
conversation_history = []

output_file = "accounting_records.jsonl"

# Definition of the expected format with Pydantics
class Buchungssatz(BaseModel):
    frage: str
    typ: str

# Function to create the prompt
def build_prompt():
    types = [
        """Geschäftsvorfälle mit internationalen Bezügen
        Beschreibung:
        Internationale Geschäftsvorfälle beinhalten häufig die Abwicklung von Importen und Exporten, welche durch Währungsumrechnungen, Zollabgaben und Einfuhrumsatzsteuer komplex werden. Herausforderungen entstehen auch durch unterschiedliche Rechtsvorschriften, wie das Zollrecht oder Rechnungslegungsvorgaben nach IFRS und HGB. Fehler können schnell zu falscher Umsatzsteuerberechnung oder unzureichender Kostenerfassung führen.

        Beispiel:
        Ein Unternehmen kauft Waren aus den USA und bezahlt in US-Dollar. Neben dem Kaufpreis fallen Zölle, Frachtkosten und Einfuhrumsatzsteuer an.

        Wonach man googlen sollte:

        „Buchungssätze Importgeschäft“
        „Einfuhrumsatzsteuer buchen HGB“
        „Währungsumrechnung Buchhaltung Beispiel“
        „Zollkosten Kontenrahmen SKR03/SKR04“
        „GoBD Anforderungen internationale Rechnungen“""",
        """Abbildung von Rückstellungen
        Beschreibung:
        Rückstellungen dienen der Erfassung von unsicheren Verbindlichkeiten oder drohenden Verpflichtungen in der Bilanz. Ihre Bewertung ist oft schwierig, da sie auf Schätzungen basiert. Auch steuerrechtlich unterscheiden sich bilanzielle und steuerliche Rückstellungen. Häufig sind nicht alle gebildeten Rückstellungen steuerlich abzugsfähig.

        Beispiel:
        Ein Unternehmen erwartet Prozesskosten aufgrund eines laufenden Rechtsstreits und muss hierfür eine Rückstellung bilden.

        Wonach man googlen sollte:

        „Rückstellungen buchen Beispiel HGB“
        „Steuerlich abzugsfähige Rückstellungen EStG“
        „Bewertung von Rückstellungen nach GoB“
        „Rückstellungen Betriebsprüfung Besonderheiten“""",
        """Sonderfälle bei der Umsatzsteuer
        Beschreibung:
        Sonderfälle bei der Umsatzsteuer betreffen z. B. innergemeinschaftliche Lieferungen, Steuerbefreiungen oder das Reverse-Charge-Verfahren. Diese Vorgänge erfordern eine korrekte Zuordnung der Umsatzsteuer und die Einhaltung bestimmter Dokumentationspflichten, da Verstöße zu erheblichen Nachzahlungen führen können.

        Beispiel:
        Ein Unternehmer erbringt eine Dienstleistung an ein ausländisches Unternehmen innerhalb der EU. Hier gilt das Reverse-Charge-Verfahren, sodass der Leistungsempfänger die Umsatzsteuer schuldet.

        Wonach man googlen sollte:

        „Reverse Charge Buchungssatz Beispiel“
        „Innergemeinschaftliche Lieferung Kontenrahmen SKR03“
        „Umsatzsteuer Steuerbefreiung § 4 UStG“
        „Umsatzsteuer Sonderfälle Handbuch“""",
        """Nicht abzugsfähige Betriebsausgaben
        Beschreibung:
        § 4 Abs. 5 EStG definiert bestimmte Betriebsausgaben, die steuerlich nicht abzugsfähig sind. Diese müssen in der Buchhaltung sauber getrennt werden, da sie nicht die Steuerlast mindern dürfen. Dies betrifft z. B. Bewirtungskosten oder Geschenke über 35 Euro.

        Beispiel:
        Ein Unternehmen lädt Geschäftspartner zum Essen ein. Die Kosten betragen 500 Euro. 70% der Bewirtungskosten sind abzugsfähig, 30% hingegen nicht.

        Wonach man googlen sollte:

        „Nicht abzugsfähige Betriebsausgaben Liste EStG“
        „Bewirtungskosten buchen SKR03/SKR04“
        „Betriebsausgaben steuerlich abziehbar Übersicht“
        „Buchungssätze nicht abzugsfähige Kosten Beispiele“""",
        """Bilanzierung immaterieller Vermögenswerte
        Beschreibung:
        Immaterielle Vermögenswerte wie Patente, Marken oder selbst erstellte Software stellen eine besondere Herausforderung dar, da ihre Bewertung oft unklar ist. Die Unterscheidung zwischen aktivierungspflichtigen und laufenden Kosten ist entscheidend, um Fehler bei der Bilanzierung zu vermeiden.

        Beispiel:
        Ein Unternehmen entwickelt eine Software intern und möchte die Entwicklungskosten aktivieren. Es muss dabei zwischen Forschungs- und Entwicklungskosten differenzieren.

        Wonach man googlen sollte:

        „Immaterielle Vermögenswerte HGB Beispiele“
        „Aktivierungspflicht selbst geschaffener Software“
        „Forschungs- und Entwicklungskosten Bilanzierung“
        „Abschreibung immaterielle Vermögenswerte Steuerrecht“""",
        """Verlustvortrag und -rücktrag
        Beschreibung:
        Verluste können gemäß § 10d EStG vorgetragen oder rückgetragen werden. Dies erfordert eine exakte buchhalterische Erfassung und steuerrechtliche Abgrenzung, da Verluste nur innerhalb bestimmter Grenzen genutzt werden können (z. B. Mindestbesteuerung).

        Beispiel:
        Ein Unternehmen erzielt 2023 einen Verlust von 100.000 Euro. Dieser wird mit dem Gewinn aus 2024 verrechnet.

        Wonach man googlen sollte:

        „Verlustvortrag buchen EStG“
        „Verlustverrechnung steuerliche Behandlung“
        „Mindestbesteuerung Verlustvortrag Berechnung“
        „Buchung Verlustausgleich SKR03“""",
        """Buchung von Investitionsabzugsbeträgen (IAB)
        Beschreibung:
        Investitionsabzugsbeträge (§ 7g EStG) erlauben Steuerersparnisse für geplante Investitionen. Die Buchung der IAB und deren Auflösung erfordern eine genaue Dokumentation, da sie an strenge Voraussetzungen geknüpft sind (z. B. Betriebsgrößengrenzen).

        Beispiel:
        Ein Unternehmen plant den Kauf eines Fahrzeugs und bildet 2024 einen IAB von 40%.

        Wonach man googlen sollte:

        „Investitionsabzugsbetrag Beispiel SKR03“
        „IAB buchen und auflösen EStG“
        „Voraussetzungen für Investitionsabzugsbetrag“
        „Investitionsabzugsbetrag Checkliste“""",
        """Privatanteile und Entnahmen
        Beschreibung:
        Geschäftsvorfälle mit gemischter Nutzung, wie die private Nutzung eines Firmenwagens, erfordern eine präzise Aufteilung zwischen privaten und betrieblichen Anteilen. Dies ist oft mit zusätzlichen Nachweispflichten verbunden, z. B. einem Fahrtenbuch.

        Beispiel:
        Ein Firmenwagen wird zu 30% privat genutzt. Der Privatanteil muss korrekt verbucht werden, da er als fiktive Einnahme gilt.

        Wonach man googlen sollte:

        „Privatnutzung Firmenwagen buchen Beispiel“
        „Fahrtenbuch vs. 1%-Regelung steuerliche Unterschiede“
        „Privatanteile Buchhaltung SKR04“
        „Entnahme von Wirtschaftsgütern Steuerrecht“""",
        """Konsolidierung in der Konzernbuchhaltung
        Beschreibung:
        Die Konsolidierung in Konzernen erfordert die Abstimmung und Eliminierung konzerninterner Geschäftsvorfälle, z. B. Verrechnungspreise oder interne Warenlieferungen. Außerdem müssen latente Steuern berücksichtigt werden.

        Beispiel:
        Eine deutsche Muttergesellschaft liefert Waren an eine Tochtergesellschaft in Frankreich. Die Transaktion muss im Konzernabschluss eliminiert werden.

        Wonach man googlen sollte:

        „Konsolidierung Konzernabschluss Buchungssätze“
        „Intercompany-Verrechnungen eliminieren Beispiele“
        „Latente Steuern Berechnung HGB“
        „Verrechnungspreise Steuerrecht EU“""",
        """Korrekturen von Fehlbuchungen
        Beschreibung:
        Fehlbuchungen entstehen häufig durch Unachtsamkeit oder Unkenntnis und erfordern eine nachvollziehbare Korrektur gemäß GoBD. Dabei müssen Fehler ordnungsgemäß dokumentiert und korrigiert werden, um die Richtigkeit der Buchführung zu gewährleisten.

        Beispiel:
        Eine Eingangsrechnung wurde versehentlich auf dem falschen Konto gebucht und muss berichtigt werden.

        Wonach man googlen sollte:

        „Fehlbuchung korrigieren GoBD Beispiel“
        „Stornobuchung vs. Umbuchung Buchhaltung“
        „Buchführung Korrektur Anforderungen GoBD“
        „Fehlerhafte Umsatzsteuerbuchung bereinigen“""",
        """Aktive und passive Rechnungsabgrenzungsposten
        Beschreibung: Aktive und passive Rechnungsabgrenzungsposten sind eine wichtige buchhalterische Maßnahme zur periodengerechten Zuordnung von Aufwendungen und Erträgen. Sie kommen vor, wenn Zahlungen für Leistungen oder Lieferungen bereits in einer Periode erfolgen, jedoch erst in einer späteren Periode wirtschaftlich wirksam werden. Diese müssen korrekt erfasst werden, um den Grundsatz der periodengerechten Abgrenzung gemäß den GoB (Grundsätzen ordnungsmäßiger Buchführung) zu wahren.

        Aktive Rechnungsabgrenzungsposten (ARAP) werden gebildet, wenn ein Unternehmen im Voraus Zahlungen erhält oder Aufwand tätigt, der sich auf eine zukünftige Periode bezieht. Diese Zahlungen müssen als Vermögenswert in der Bilanz erfasst werden.
        Passive Rechnungsabgrenzungsposten (PRAP) entstehen, wenn bereits eine Zahlung in der aktuellen Periode erfolgt, die jedoch Aufwand oder Ertrag für eine zukünftige Periode darstellt. PRAP müssen als Verbindlichkeit in der Bilanz erfasst werden.
        Beispiel: Ein Unternehmen bezahlt im Dezember eine Jahresversicherung für das kommende Jahr. Der Betrag wird als aktiver Rechnungsabgrenzungsposten erfasst, da der Aufwand erst in den kommenden Monaten entsteht.

        Ein anderes Beispiel ist, wenn ein Unternehmen im Voraus Miete für das kommende Jahr bezahlt hat. Diese Zahlung wird als passiver Rechnungsabgrenzungsposten behandelt, da sie Erträge für die kommenden Monate betrifft.

        Wonach man googlen sollte:

        „Aktive Rechnungsabgrenzungsposten Buchungssatz“
        „Passive Rechnungsabgrenzungsposten buchen HGB“
        „Rechnungsabgrenzung nach GoB“
        „Buchung von RAP gemäß HGB“
        „Rechnungsabgrenzung Steuerrecht EStG“
        „Prüfung von Rechnungsabgrenzungsposten Betriebsprüfung“""",
        """Umwandlungen nach dem Umwandlungsrecht
        Beschreibung: Umwandlungen sind komplexe gesellschaftsrechtliche Vorgänge, bei denen Unternehmen ihre Rechtsform ändern, verschmelzen, spalten oder Vermögensübertragungen vornehmen können. Ziel dieser Maßnahmen ist es, organisatorische, steuerliche oder rechtliche Vorteile zu erreichen, beispielsweise eine effizientere Unternehmensstruktur oder eine Anpassung an veränderte wirtschaftliche Gegebenheiten. Umwandlungen müssen im Einklang mit den Vorschriften des Umwandlungsgesetzes (UmwG) erfolgen und berücksichtigen steuerrechtliche Regelungen, insbesondere im Hinblick auf die Buchwertfortführung und die Besteuerung stiller Reserven.

        Eine Verschmelzung liegt vor, wenn zwei Unternehmen zu einer neuen Einheit zusammengeführt werden oder eines in das andere aufgenommen wird. Eine Spaltung tritt ein, wenn ein Unternehmen Teile seines Vermögens auf andere Rechtsträger überträgt, entweder durch Neugründung oder Übertragung auf bestehende Gesellschaften. Auch Formwechsel, bei denen lediglich die Rechtsform des Unternehmens angepasst wird (z. B. von einer GmbH in eine AG), fallen unter das Umwandlungsrecht.

        Beispiel: Eine GmbH wird in eine AG umgewandelt, um leichter Kapital über die Börse aufzunehmen. Dabei wird das Eigenkapital der GmbH in Aktienkapital umgewandelt, und die Gesellschaft wird unter Wahrung der Identität fortgeführt.

        Ein anderes Beispiel ist die Spaltung einer AG, bei der ein Geschäftsbereich auf eine neu gegründete GmbH übertragen wird. Hierbei wird die Steuerneutralität der Spaltung gemäß § 15 UmwStG sichergestellt, sofern die Voraussetzungen wie die Buchwertfortführung erfüllt sind.

        Wonach man googlen sollte:

        „Umwandlung nach Umwandlungsgesetz Beispiele“
        „Formwechsel Steuerrecht Umwandlungssteuerrecht“
        „Buchwertfortführung nach UmwStG“
        „Verschmelzung steuerlich neutral gestalten“
        „Spaltung einer GmbH rechtliche Voraussetzungen“
        „Umwandlung Betriebsprüfung Schwerpunkte“"""
    ]

    type_selected = random.choice(types)

    base_prompt = f"""
Du bist ein Steuerrechtsexperte und hilfst mir, neue Buchungssätze zu komplexen steuerrechtlichen Fällen zu generieren. 
Berücksichtige die Historie und stelle sicher, dass jede Frage einzigartig ist. Erstelle zu folgendem Thema etwas:

{type_selected}

Formatierungsanforderung:

    Formatiere immer im folgenden Stil.
    [
        {{"frage": "Hier die Frage", "typ": "{type_selected}"}},
        ...
    ]
"""

    base_prompt += "Erstelle ein kreatives aber realistisches Szenario bei einem meiner Mandanten (ich bin Steuerberater), bei welchem durch einen Geschäftsvorfall ein Buchungssatz anfällt. Schreibe es so als wäre es eine Frage von mir an ein LLM. Spreche in der Frage niemanden direkt an, kein Hallo kein Du etc. Gehe zuerst auf das Szenario ein frage am Ende dann nach dem Buchungssatz. Mach die Frage sehr Ausführlich und versuche alle Fragetypen gleich häufig zu generieren. Nutze einen kreativen Einstieg in die Frage, nicht immer nur etwas wie 'Ein Mandat von mir' oder ähnliches. Antworte niemals mit Sätzen wie z.B. 'Es ist ratsam, einen Steuerberater oder Wirtschaftsprüfer zu konsultieren, um sicherzustellen, dass alle rechtlichen und steuerlichen Anforderungen erfüllt sind.', schließlich soll die Antwort von einem Steuerberater sein."
    base_prompt += "Nutze nicht immer das Beispiel eines Software-Unternehmens oder eines IT-Mandanten, sondern versuche mehr alle Unternehmensbereiche abzudecken, egal ob Dienstleister oder produzierendes Gewerbe, egal ob Industrie oder nicht."
    base_prompt += "Füge nie 'wo nach man googlen' sollte an oder ähnliches!" 
    messages = [
        {"role": "user", "content": base_prompt},
    ]
    return messages

def get_embedding(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Get embeddings for a list of texts using the embedding API."""
    embeddings = []
    
    # Process in batches to avoid overwhelming the API
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        try:
            response = requests.post(
                "http://localhost:8000/embed",
                json={"texts": [str(text) for text in batch_texts]},  
                timeout=50
            )
            response.raise_for_status()
            batch_embeddings = response.json().get("embeddings", [])
            if batch_embeddings:
                embeddings.extend(batch_embeddings)
            else:
                print(f"No embeddings returned for batch {i}")
                return np.zeros((len(texts), 1024))
        except Exception as e:
            print(f"Error getting embeddings for batch {i}: {e}")
            return np.zeros((len(texts), 1024))
    
    if not embeddings:
        return np.zeros((len(texts), 1024))
        
    return np.array(embeddings)

def generate_new_query(query: str):
            # Prepare messages for chat completion
        messages = [
            {"role": "system", "content": "Du erhältst immer eine Frage in Form einer alten Suchmaschinen-Query, die zu keinem Ergebnis geführt hat. Deine Aufgabe ist es, basierend auf den wichtigen Aspekten der alten Query eine neue Such-Query zu erstellen, die sich auf steuerrechtliche Fragestellungen konzentriert und für die Suchmaschine besser verständlich ist. Extrahiere die relevanten Informationen aus der ursprünglichen Query und optimiere die neue Query so, dass sie steuerrechtliche Aspekte klar hervorhebt. Gib ausschließlich die neue Query zurück, ohne zusätzliche Kommentare, Erläuterungen oder Formatierungen. Beispiel: Alter Kontext: 'Die Bäcker- & Konditormeister Müller und Weber interessieren sich für einen Volkswagen Transporter, den sie als Auslieferungswagen anschaffen möchten. Da sie den Wagen nicht bar bezahlen können, schlägt Ihnen der Volkswagen-Händler eine entsprechende Kredit- oder Leasingfinanzierung vor. Erläutern Sie bitte die wesentlichen Bedingungen eines Leasingvertrages.' Neue Query: 'steuerrechtliche Behandlung Leasingvertrag'"},
            {"role": "system", "content": """
Hier hast du ein paar knifflige Fälle und Beispiele was man googlen kann.
1. Geschäftsvorfälle mit internationalen Bezügen
        Beschreibung:
        Internationale Geschäftsvorfälle beinhalten häufig die Abwicklung von Importen und Exporten, welche durch Währungsumrechnungen, Zollabgaben und Einfuhrumsatzsteuer komplex werden. Herausforderungen entstehen auch durch unterschiedliche Rechtsvorschriften, wie das Zollrecht oder Rechnungslegungsvorgaben nach IFRS und HGB. Fehler können schnell zu falscher Umsatzsteuerberechnung oder unzureichender Kostenerfassung führen.

        Beispiel:
        Ein Unternehmen kauft Waren aus den USA und bezahlt in US-Dollar. Neben dem Kaufpreis fallen Zölle, Frachtkosten und Einfuhrumsatzsteuer an.

        Wonach man googlen sollte:

        „Buchungssätze Importgeschäft“
        „Einfuhrumsatzsteuer buchen HGB“
        „Währungsumrechnung Buchhaltung Beispiel“
        „Zollkosten Kontenrahmen SKR03/SKR04“
        „GoBD Anforderungen internationale Rechnungen“

2. Abbildung von Rückstellungen
        Beschreibung:
        Rückstellungen dienen der Erfassung von unsicheren Verbindlichkeiten oder drohenden Verpflichtungen in der Bilanz. Ihre Bewertung ist oft schwierig, da sie auf Schätzungen basiert. Auch steuerrechtlich unterscheiden sich bilanzielle und steuerliche Rückstellungen. Häufig sind nicht alle gebildeten Rückstellungen steuerlich abzugsfähig.

        Beispiel:
        Ein Unternehmen erwartet Prozesskosten aufgrund eines laufenden Rechtsstreits und muss hierfür eine Rückstellung bilden.

        Wonach man googlen sollte:

        „Rückstellungen buchen Beispiel HGB“
        „Steuerlich abzugsfähige Rückstellungen EStG“
        „Bewertung von Rückstellungen nach GoB“
        „Rückstellungen Betriebsprüfung Besonderheiten“

3. Sonderfälle bei der Umsatzsteuer
        Beschreibung:
        Sonderfälle bei der Umsatzsteuer betreffen z. B. innergemeinschaftliche Lieferungen, Steuerbefreiungen oder das Reverse-Charge-Verfahren. Diese Vorgänge erfordern eine korrekte Zuordnung der Umsatzsteuer und die Einhaltung bestimmter Dokumentationspflichten, da Verstöße zu erheblichen Nachzahlungen führen können.

        Beispiel:
        Ein Unternehmer erbringt eine Dienstleistung an ein ausländisches Unternehmen innerhalb der EU. Hier gilt das Reverse-Charge-Verfahren, sodass der Leistungsempfänger die Umsatzsteuer schuldet.

        Wonach man googlen sollte:

        „Reverse Charge Buchungssatz Beispiel“
        „Innergemeinschaftliche Lieferung Kontenrahmen SKR03“
        „Umsatzsteuer Steuerbefreiung § 4 UStG“
        „Umsatzsteuer Sonderfälle Handbuch“

4. Nicht abzugsfähige Betriebsausgaben
        Beschreibung:
        § 4 Abs. 5 EStG definiert bestimmte Betriebsausgaben, die steuerlich nicht abzugsfähig sind. Diese müssen in der Buchhaltung sauber getrennt werden, da sie nicht die Steuerlast mindern dürfen. Dies betrifft z. B. Bewirtungskosten oder Geschenke über 35 Euro.

        Beispiel:
        Ein Unternehmen lädt Geschäftspartner zum Essen ein. Die Kosten betragen 500 Euro. 70% der Bewirtungskosten sind abzugsfähig, 30% hingegen nicht.

        Wonach man googlen sollte:

        „Nicht abzugsfähige Betriebsausgaben Liste EStG“
        „Bewirtungskosten buchen SKR03/SKR04“
        „Betriebsausgaben steuerlich abziehbar Übersicht“
        „Buchungssätze nicht abzugsfähige Kosten Beispiele“

5. Bilanzierung immaterieller Vermögenswerte
        Beschreibung:
        Immaterielle Vermögenswerte wie Patente, Marken oder selbst erstellte Software stellen eine besondere Herausforderung dar, da ihre Bewertung oft unklar ist. Die Unterscheidung zwischen aktivierungspflichtigen und laufenden Kosten ist entscheidend, um Fehler bei der Bilanzierung zu vermeiden.

        Beispiel:
        Ein Unternehmen entwickelt eine Software intern und möchte die Entwicklungskosten aktivieren. Es muss dabei zwischen Forschungs- und Entwicklungskosten differenzieren.

        Wonach man googlen sollte:

        „Immaterielle Vermögenswerte HGB Beispiele“
        „Aktivierungspflicht selbst geschaffener Software“
        „Forschungs- und Entwicklungskosten Bilanzierung“
        „Abschreibung immaterielle Vermögenswerte Steuerrecht“

6. Verlustvortrag und -rücktrag
        Beschreibung:
        Verluste können gemäß § 10d EStG vorgetragen oder rückgetragen werden. Dies erfordert eine exakte buchhalterische Erfassung und steuerrechtliche Abgrenzung, da Verluste nur innerhalb bestimmter Grenzen genutzt werden können (z. B. Mindestbesteuerung).

        Beispiel:
        Ein Unternehmen erzielt 2023 einen Verlust von 100.000 Euro. Dieser wird mit dem Gewinn aus 2024 verrechnet.

        Wonach man googlen sollte:

        „Verlustvortrag buchen EStG“
        „Verlustverrechnung steuerliche Behandlung“
        „Mindestbesteuerung Verlustvortrag Berechnung“
        „Buchung Verlustausgleich SKR03“

7. Buchung von Investitionsabzugsbeträgen (IAB)
        Beschreibung:
        Investitionsabzugsbeträge (§ 7g EStG) erlauben Steuerersparnisse für geplante Investitionen. Die Buchung der IAB und deren Auflösung erfordern eine genaue Dokumentation, da sie an strenge Voraussetzungen geknüpft sind (z. B. Betriebsgrößengrenzen).

        Beispiel:
        Ein Unternehmen plant den Kauf eines Fahrzeugs und bildet 2024 einen IAB von 40%.

        Wonach man googlen sollte:

        „Investitionsabzugsbetrag Beispiel SKR03“
        „IAB buchen und auflösen EStG“
        „Voraussetzungen für Investitionsabzugsbetrag“
        „Investitionsabzugsbetrag Checkliste“

8. Privatanteile und Entnahmen
        Beschreibung:
        Geschäftsvorfälle mit gemischter Nutzung, wie die private Nutzung eines Firmenwagens, erfordern eine präzise Aufteilung zwischen privaten und betrieblichen Anteilen. Dies ist oft mit zusätzlichen Nachweispflichten verbunden, z. B. einem Fahrtenbuch.

        Beispiel:
        Ein Firmenwagen wird zu 30% privat genutzt. Der Privatanteil muss korrekt verbucht werden, da er als fiktive Einnahme gilt.

        Wonach man googlen sollte:

        „Privatnutzung Firmenwagen buchen Beispiel“
        „Fahrtenbuch vs. 1%-Regelung steuerliche Unterschiede“
        „Privatanteile Buchhaltung SKR04“
        „Entnahme von Wirtschaftsgütern Steuerrecht“

9. Konsolidierung in der Konzernbuchhaltung
        Beschreibung:
        Die Konsolidierung in Konzernen erfordert die Abstimmung und Eliminierung konzerninterner Geschäftsvorfälle, z. B. Verrechnungspreise oder interne Warenlieferungen. Außerdem müssen latente Steuern berücksichtigt werden.

        Beispiel:
        Eine deutsche Muttergesellschaft liefert Waren an eine Tochtergesellschaft in Frankreich. Die Transaktion muss im Konzernabschluss eliminiert werden.

        Wonach man googlen sollte:

        „Konsolidierung Konzernabschluss Buchungssätze“
        „Intercompany-Verrechnungen eliminieren Beispiele“
        „Latente Steuern Berechnung HGB“
        „Verrechnungspreise Steuerrecht EU“
        10. Korrekturen von Fehlbuchungen
        Beschreibung:
        Fehlbuchungen entstehen häufig durch Unachtsamkeit oder Unkenntnis und erfordern eine nachvollziehbare Korrektur gemäß GoBD. Dabei müssen Fehler ordnungsgemäß dokumentiert und korrigiert werden, um die Richtigkeit der Buchführung zu gewährleisten.

        Beispiel:
        Eine Eingangsrechnung wurde versehentlich auf dem falschen Konto gebucht und muss berichtigt werden.

        Wonach man googlen sollte:

        „Fehlbuchung korrigieren GoBD Beispiel“
        „Stornobuchung vs. Umbuchung Buchhaltung“
        „Buchführung Korrektur Anforderungen GoBD“
        „Fehlerhafte Umsatzsteuerbuchung bereinigen“

11. Aktive und passive Rechnungsabgrenzungsposten
        Beschreibung: Aktive und passive Rechnungsabgrenzungsposten sind eine wichtige buchhalterische Maßnahme zur periodengerechten Zuordnung von Aufwendungen und Erträgen. Sie kommen vor, wenn Zahlungen für Leistungen oder Lieferungen bereits in einer Periode erfolgen, jedoch erst in einer späteren Periode wirtschaftlich wirksam werden. Diese müssen korrekt erfasst werden, um den Grundsatz der periodengerechten Abgrenzung gemäß den GoB (Grundsätzen ordnungsmäßiger Buchführung) zu wahren.

        Aktive Rechnungsabgrenzungsposten (ARAP) werden gebildet, wenn ein Unternehmen im Voraus Zahlungen erhält oder Aufwand tätigt, der sich auf eine zukünftige Periode bezieht. Diese Zahlungen müssen als Vermögenswert in der Bilanz erfasst werden.
        Passive Rechnungsabgrenzungsposten (PRAP) entstehen, wenn bereits eine Zahlung in der aktuellen Periode erfolgt, die jedoch Aufwand oder Ertrag für eine zukünftige Periode darstellt. PRAP müssen als Verbindlichkeit in der Bilanz erfasst werden.
        Beispiel: Ein Unternehmen bezahlt im Dezember eine Jahresversicherung für das kommende Jahr. Der Betrag wird als aktiver Rechnungsabgrenzungsposten erfasst, da der Aufwand erst in den kommenden Monaten entsteht.

        Ein anderes Beispiel ist, wenn ein Unternehmen im Voraus Miete für das kommende Jahr bezahlt hat. Diese Zahlung wird als passiver Rechnungsabgrenzungsposten behandelt, da sie Erträge für die kommenden Monate betrifft.

        Wonach man googlen sollte:

        „Aktive Rechnungsabgrenzungsposten Buchungssatz“
        „Passive Rechnungsabgrenzungsposten buchen HGB“
        „Rechnungsabgrenzung nach GoB“
        „Buchung von RAP gemäß HGB“
        „Rechnungsabgrenzung Steuerrecht EStG“
        „Prüfung von Rechnungsabgrenzungsposten Betriebsprüfung
             
12. Umwandlungen nach dem Umwandlungsrecht
        Beschreibung: Umwandlungen sind komplexe gesellschaftsrechtliche Vorgänge, bei denen Unternehmen ihre Rechtsform ändern, verschmelzen, spalten oder Vermögensübertragungen vornehmen können. Ziel dieser Maßnahmen ist es, organisatorische, steuerliche oder rechtliche Vorteile zu erreichen, beispielsweise eine effizientere Unternehmensstruktur oder eine Anpassung an veränderte wirtschaftliche Gegebenheiten. Umwandlungen müssen im Einklang mit den Vorschriften des Umwandlungsgesetzes (UmwG) erfolgen und berücksichtigen steuerrechtliche Regelungen, insbesondere im Hinblick auf die Buchwertfortführung und die Besteuerung stiller Reserven.

        Eine Verschmelzung liegt vor, wenn zwei Unternehmen zu einer neuen Einheit zusammengeführt werden oder eines in das andere aufgenommen wird. Eine Spaltung tritt ein, wenn ein Unternehmen Teile seines Vermögens auf andere Rechtsträger überträgt, entweder durch Neugründung oder Übertragung auf bestehende Gesellschaften. Auch Formwechsel, bei denen lediglich die Rechtsform des Unternehmens angepasst wird (z. B. von einer GmbH in eine AG), fallen unter das Umwandlungsrecht.

        Beispiel: Eine GmbH wird in eine AG umgewandelt, um leichter Kapital über die Börse aufzunehmen. Dabei wird das Eigenkapital der GmbH in Aktienkapital umgewandelt, und die Gesellschaft wird unter Wahrung der Identität fortgeführt.

        Ein anderes Beispiel ist die Spaltung einer AG, bei der ein Geschäftsbereich auf eine neu gegründete GmbH übertragen wird. Hierbei wird die Steuerneutralität der Spaltung gemäß § 15 UmwStG sichergestellt, sofern die Voraussetzungen wie die Buchwertfortführung erfüllt sind.

        Wonach man googlen sollte:

        „Umwandlung nach Umwandlungsgesetz Beispiele“
        „Formwechsel Steuerrecht Umwandlungssteuerrecht“
        „Buchwertfortführung nach UmwStG“
        „Verschmelzung steuerlich neutral gestalten“
        „Spaltung einer GmbH rechtliche Voraussetzungen“
        „Umwandlung Betriebsprüfung Schwerpunkte“"""},
            {"role": "user", "content": f"Kontext:\n{query}\n Frage: Erstelle mir dazu eine neue Query für Search Enginges und fokussiere dich dabei darauf, dass es um das Bilanzierung und Buchführung bzw. Buchungssätze geht. Gebe mir nur die neue query aus und sonst gar nichts. Die Query muss immer auf Deutsch sein!"}
        ]
        
        # Make API call with retry logic
        response = chat(
            messages=messages,
            model="llama3:8b-instruct-q8_0 ", # Needs to be adjusted
        )
        if "message" in response and "content" in response["message"]:
            raw_content = response["message"]["content"]
            print("Raw:", raw_content)
        return raw_content

def scrape_and_process_url(url: str) -> str:
    """Scrape content from a URL and process it (HTML, XML, or PDF)."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=180)
        response.raise_for_status()

        # Check for PDF
        content_type = response.headers.get('Content-Type', '')
        if 'application/pdf' in content_type:
            pdf_content = BytesIO(response.content)
            try:
                reader = PdfReader(pdf_content)

                # Extract text from each page
                text = "\n".join(
                    page.extract_text() for page in reader.pages if page.extract_text()
                )
            except PdfReadError as e:
                
                return ""

        # Check for XML or HTML
        elif url.endswith(".xml"):
            soup = BeautifulSoup(response.content, 'xml')
            text = soup.get_text(separator=' ', strip=True)
        else:
            soup = BeautifulSoup(response.text, 'html.parser')

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text content
            text = soup.get_text(separator=' ', strip=True)

        # Clean up whitespace
        text = ' '.join(text.split())
        return text

    except Exception as e:
        
        return ""

def chunk_text(text: str, num_chunks: int = 3) -> List[str]:
    """Split text into exactly num_chunks chunks of approximately equal size."""
    # Split into sentences
    sentences = sent_tokenize(text)
    if len(sentences) < num_chunks:
        return sentences  # Return all sentences if fewer than num_chunks
        
    # Calculate target size for each chunk
    total_length = sum(len(s) for s in sentences)
    target_chunk_size = total_length // num_chunks
    
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_count = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)
        
        # Check if we should create a new chunk
        if current_length >= target_chunk_size and chunk_count < num_chunks - 1:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
            chunk_count += 1
    
    # Add remaining sentences to the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def get_searxng_results(query: str, top_k: int = 3):
    """Get search results from SearXNG, scrape results, and calculate similarity scores."""
    try:
        params = {
            "q": query,
            "format": "json",
            "engines": "wiki",
            "language": "de",
            "max_results": 6
        }

        # Make request to SearXNG
        results = make_searxng_request(searxng_url, params)
        
        # Get query embedding
        query_embedding = get_embedding([query])
        
        processed_results = []

        counter = 0
        
        # Process only the first `top_k` results
        for idx, result in enumerate(results.get("results", [])[:top_k]):
            url = result.get('url', '')
            scraped_content = scrape_and_process_url(url)
            
            if scraped_content == "":
                counter += 1
                continue
            
            chunks = chunk_text(scraped_content)
            
            for chunk in chunks:
                chunk_embedding = get_embedding([chunk])[0]
                similarity = cosine_similarity(query_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0]

                processed_results.append({
                    "score": float(similarity),
                    "article": {
                        "url": url,
                        "title": result.get('title', ''),
                        "content": chunk
                    },
                    "source": "SearXNG"
            })
        
        # Process remaining results if some were skipped
        if counter > 0:
            end = top_k + counter
            for idx, result in enumerate(results.get("results", [])[top_k:end]):
                url = result.get('url', '')
                scraped_content = scrape_and_process_url(url)
                
                if scraped_content == "":
                    continue
                
                chunks = chunk_text(scraped_content)
                
                for chunk in chunks:
                    chunk_embedding = get_embedding([chunk], 240)[0]
                    similarity = cosine_similarity(query_embedding.reshape(1, -1), chunk_embedding.reshape(1, -1))[0][0]

                    processed_results.append({
                        "score": float(similarity),
                        "article": {
                            "url": url,
                            "title": result.get('title', ''),
                            "content": chunk
                        },
                        "source": "SearXNG"
                    })
        
        # If no valid results, retry with a new query
        if not processed_results:
            query = generate_new_query(query)
            processed_results = get_searxng_results(query, top_k)
        
        return processed_results

    except Exception as e:
        print(e)
        return []

def merge_and_rank_results(searxng_results: List[dict], top_k: int = 20) -> List[dict]:
    """Merge and rank results from both sources."""
    ranked_results = sorted(searxng_results, key=lambda x: x["score"], reverse=True)
    return ranked_results[:top_k]

def get_relevant_texts(results: List[Dict], max_token_count: int = 30000) -> List[Dict]:
    """Extract the most relevant text passages from results up to max_token_count total."""
    relevant_texts = []
    total_tokens = 0
  
    for result in results:
        
        article = result["article"]
        text = article.get('content', '')
        source_info = {
                "source": "SearXNG",
                "url": article.get('url', ''),
                "title": article.get('title', '')
            }
        
        # Count tokens using the API
        try:
            response = requests.post(
                "http://localhost:8001/count_tokens",
                json={"text": text},
                timeout=50
            )
            response.raise_for_status()
            token_count = response.json()["count"]
        except Exception as e:
            print(f"Error counting tokens: {e}")
            continue
        
        # Check if adding this text would exceed the token limit
        if total_tokens + token_count > max_token_count and token_count < 15000:
            # Calculate remaining tokens
            remaining_tokens = max_token_count - total_tokens
            if remaining_tokens > 0:
                try:
                    # Truncate the text using the API
                    response = requests.post(
                        "http://localhost:8001/tokenize",
                        json={
                            "texts": [text],
                            "max_length": remaining_tokens,
                            "truncation": True
                        },
                        timeout=50
                    )
                    response.raise_for_status()
                    truncated_text = response.json()["texts"][0]
                    relevant_texts.append({
                        "text": truncated_text,
                        "score": result["score"],
                        "source_info": source_info
                    })
                except Exception as e:
                    print(f"Error truncating text: {e}")
            break
        
        if token_count < 30000:
            relevant_texts.append({
                "text": text,
                "score": result["score"],
                "source_info": source_info
            })
            total_tokens += token_count
    
    return relevant_texts

def make_searxng_request(url: str, params: dict) -> dict:
    """Make a request to SearXNG with retry logic."""
    response = requests.get(url, headers={"User-Agent": "Legal-Retrieval-Bot"}, params=params)
    response.raise_for_status()
    return response.json()

def save_to_file(question, answer):
    entry = {
        "frage": question,
        "antwort": answer
    }
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")  # Save each answer in a new lines
    except Exception as e:
        print(f"Fehler beim Speichern der Antwort in der Datei: {e}")

def get_answer(context_texts):
    try:
        context_parts = []
        for ctx in context_texts:
            source_info = ctx["source_info"]

            source_str = f"[Source: {source_info['title']} - {source_info['url']}]"
        
            
            context_parts.append(f"{source_str}\n{ctx['text']}\n")
        
        context = "\n".join(context_parts)

        # Prepare messages for chat completion
        messages = [
            {"role": "user", "content": f"Kontext:\n{context}\n\nFrage: {query}\n\n. Beantworte die Frage und bilde den Buchungssatz."}
        ]
        output = llm.beta.chat.completions.parse(
            model=AZURE_DEPLOYMENT_ID,
            messages=messages,
            temperature=0.2,
            top_p=0.9
        )

        # Check whether the answer structure is correct
        if output and output.choices and output.choices[0].message:
            message = output.choices[0].message
            if message.content:
                return message.content
                
            else:
                
                return
        else:
            
            return 


    except Exception as e:
        return 
       

# Function for the request to Ollama
def generate_buchungssatz_request():
    # Prompt creation
    prompt = build_prompt()

    try:
        # Request to the model with formatting
        response = chat(
            messages=prompt,
            model="llama3:8b-instruct-q8_0 ",
            format=Buchungssatz.model_json_schema(),  # Scheme for the answer
        )

        # Process the answer
        if "message" in response and "content" in response["message"]:
            raw_content = response["message"]["content"]
            buchungssatz = Buchungssatz.model_validate_json(raw_content)
            return buchungssatz
        else:
            raise ValueError("Response structure does not contain a 'content' field.")

    except Exception as e:
        print(f"Error during enquiry or processing: {e}")
        return None

# Endless loop for generation and storage
print("Start generation of new account records...")
while True:
    # Generate a new question based on the history
    result = generate_buchungssatz_request()

    if result:
        # Check whether the question is unique
        if result.frage not in conversation_history:
            conversation_history.append(result.frage)
            print(f"Neue Frage generiert:\nFrage: {result.frage}\nTyp: {result.typ}")

            # Conversion of the question into a search query and request to endpoint
            query = generate_new_query(result.frage)

            all_results = []

            searxng_results = get_searxng_results(query=query) 
            
            # Merge all results
            all_results.extend(merge_and_rank_results(searxng_results))
            
            # Get relevant texts with max token count
            max_token_count = 35000  # Set a fixed maximum token count
            context_texts = get_relevant_texts(all_results, max_token_count)

            answer = get_answer(context_texts)

            if answer:
                print(f"Answer received: {answer}")
                # Speichern der Frage und Antwort
                save_to_file(result.frage, answer)
            else:
                print("No response received from the endpoint.")
        else:
            print("Double question recognised. New enquiry is sent...")

    else:
        print("No response received from the model. Wait...")
