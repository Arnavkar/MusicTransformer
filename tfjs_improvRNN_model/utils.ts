const maxApi = require("max-api");

export type NumBeats = 2 | 3 | 4 | 5 | 6 | 7 | 8;
export type BeatValue = 4 | 8;
export type StepsPerQuarter = 1 | 2 | 3 | 4 | 8 | 16 | 32;
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