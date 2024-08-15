import { LanguageData } from "./language-data";
import { CurrencyData } from "./currency-data";

export class CountryData{
    country_id?: number;
    country_code?: string;
    country_name?: string;
    display_name?: string;
    phone_code?: number;
    languages?: LanguageData[];
    currencies?: CurrencyData[];
}