import config from './frontend_config.json';

const BASE_URL = config.api.base_url;

/**
 * Fetches the list of available Bible translations.
 * @returns {Promise} A promise resolving to the list of Bibles.
 */
export async function listBibles() {
    const response = await fetch(`${BASE_URL}${config.api.endpoints.list_bibles}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch bibles: ${response.statusText}`);
    }
    return response.json();
}

/**
 * Fetches the content of a specific Bible translation.
 * @param {string} version - The version ID of the Bible to fetch.
 * @returns {Promise} A promise resolving to the Bible content.
 */
export async function getBible(version) {
    const endpoint = config.api.endpoints.get_bible.replace('{version}', version);
    const response = await fetch(`${BASE_URL}${endpoint}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch Bible version '${version}': ${response.statusText}`);
    }
    return response.json();
}