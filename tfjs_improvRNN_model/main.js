"use strict";
var __awaiter = (this && this.__awaiter) || function (thisArg, _arguments, P, generator) {
    function adopt(value) { return value instanceof P ? value : new P(function (resolve) { resolve(value); }); }
    return new (P || (P = Promise))(function (resolve, reject) {
        function fulfilled(value) { try { step(generator.next(value)); } catch (e) { reject(e); } }
        function rejected(value) { try { step(generator["throw"](value)); } catch (e) { reject(e); } }
        function step(result) { result.done ? resolve(result.value) : adopt(result.value).then(fulfilled, rejected); }
        step((generator = generator.apply(thisArg, _arguments || [])).next());
    });
};
Object.defineProperty(exports, "__esModule", { value: true });
const improvisor_1 = require("./improvisor");
const maxApi = require("max-api");
let improvisor;
//retrieve time Settings from Max Patch, initialize Improvisor
maxApi.addHandler("setTimeSettings", (numbeats, beatvalue, stepsPerQuarter, qpm) => {
    //measured beat is currently not used
    const timeSettings = {
        numbeats: numbeats,
        beatvalue: beatvalue,
        stepsPerQuarter: stepsPerQuarter,
        qpm: qpm
    };
    if (improvisor) {
        improvisor.updateTimeSettings(timeSettings);
    }
    else {
        improvisor = new improvisor_1.Improvisor(timeSettings);
    }
    console.log(improvisor.timeSettings);
});
maxApi.addHandler("getNotes", (...midiNotes) => {
    if (midiNotes.length == 0) {
        return;
    }
    let notes = [];
    for (let i = 0; i < midiNotes.length; i += 4) {
        let note = {
            pitch: midiNotes[i],
            velocity: midiNotes[i + 1],
            duration: midiNotes[i + 2] / 1000,
            startTime: midiNotes[i + 3] //already in seconds
        };
        if (improvisor.inputNotes === null) {
            improvisor.inputNotes = [];
        }
        notes.push(note);
    }
    improvisor.inputNotes = notes;
    console.log("INPUT NOTES–––––––––––––\n", improvisor.inputNotes);
    improvisor.quantizeInputNotes();
    console.log("QUANTIZED INPUT NOTES–––––––––––––\n", improvisor.quantizedInput);
});
maxApi.addHandler("generateSequence", () => __awaiter(void 0, void 0, void 0, function* () {
    let generated = yield improvisor.generateNewSequence();
    console.log("GENERATED NOTES–––––––––––––\n", generated);
    maxApi.outlet(generated);
}));
