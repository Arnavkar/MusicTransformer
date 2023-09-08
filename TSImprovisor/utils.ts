const maxApi = require("max-api");

export type NumBeats = 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12;
export type BeatValue = 4 | 8;
export type StepsPerQuarter = 1 | 2 | 4 | 6 | 8 | 12; //let 12 be default (whole, half, triplet half, 4th, triple 4th, 8th, triplet 8th, 16th, triplet 16th)
//Array of strings representing chords
export type ChordProg = Array<string>;
export interface Note {
    pitch: number;
    velocity: number;
    duration: number;
    startTime: number; //start time from start of the bar
}

export interface TimeSettings {
    numbeats: NumBeats
    beatvalue: BeatValue
    stepsPerQuarter:StepsPerQuarter
    qpm:number //quarter notes per minute
}

export function logErrorToMax(error: any) {
    maxApi.post(error, maxApi.POST_LEVELS.ERROR);
}