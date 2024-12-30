import {Client} from "langsmith";
import {NextRequest} from "next/server";

export const runtime = "egde";

const client = new Client();

export async function POST(req: NextRequest) {

}