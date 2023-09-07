"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.logErrorToMax = void 0;
const maxApi = require("max-api");
function logErrorToMax(error) {
    maxApi.post(error, maxApi.POST_LEVELS.ERROR);
}
exports.logErrorToMax = logErrorToMax;
