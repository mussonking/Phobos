"""GLM-family template and streaming frontend contract tests."""
from __future__ import annotations

import json
import os
import sys
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from serve.frontend import (ChatTemplate, GLMOutputParser, anthropic_to_messages,  # noqa: E402
                            openai_to_messages)
from serve.server import (GLM_STOP_IDS, ByteTokenizer, MockEngine, Service, openai_chunks,  # noqa: E402
                          openai_collect)


class GLMParser(unittest.TestCase):
    def test_streaming_parser_handles_tags_split_at_arbitrary_boundaries(self):
        tools = [{"name": "lookup", "parameters": {"properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}, "exact": {"type": "boolean"}}}}]
        parser = GLMOutputParser(thinking=True, tools=tools, stream_tools=True)
        chunks = ["Let me look.", "</thi", "nk>\n\nSearching.", "<tool_", "call>lookup<arg_ke",
                  "y>query</arg_key><arg_va", "lue>Québec</arg_value><arg_key>limit</arg_key>",
                  "<arg_value>3</arg_value><arg_key>exact</arg_key><arg_value>true</arg_value>",
                  "</tool_call> Done."]
        events = [event for chunk in chunks for event in parser.feed(chunk)] + parser.finish()
        self.assertEqual("".join(e.text for e in events if e.kind == "reasoning"), "Let me look.")
        self.assertEqual("".join(e.text for e in events if e.kind == "content"), "Searching. Done.")
        starts = [e for e in events if e.kind == "tool_start"]
        calls = [e.call for e in events if e.kind == "tool_call"]
        self.assertEqual(len(starts), 1)
        self.assertEqual(starts[0].call.name, "lookup")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].arguments, {"query": "Québec", "limit": 3, "exact": True})
        streamed = "".join(e.text for e in events if e.kind == "tool_args")
        self.assertEqual(streamed, '{"query":"Québec","limit":3,"exact":true}')

    def test_final_openai_tool_call_has_the_standard_tool_calls_shape(self):
        response = (
            "Thinking.</think>\n\n<tool_call>lookup<arg_key>query</arg_key>"
            "<arg_value>Paris</arg_value><arg_key>limit</arg_key><arg_value>2</arg_value></tool_call>"
        )

        class TextEngine:
            max_context = 4096

            def generate(self, ids, max_new, sampling, cancel, embeddings=None):
                yield from response.encode("utf-8")
                yield 154827  # GLM eot; the API maps this to finish_reason=tool_calls.

        tok = ByteTokenizer()
        service = Service(TextEngine(), tok, ChatTemplate(ROOT / "serve/glm_chat_template.jinja"), family="glm")
        tools = [{"name": "lookup", "parameters": {"properties": {
            "query": {"type": "string"}, "limit": {"type": "integer"}}}}]
        result = openai_collect(openai_chunks(service, {"model": "test"}, [ord("p")], True, tools, 100,
                                              threading.Event()))
        choice = result["choices"][0]
        call = choice["message"]["tool_calls"][0]
        self.assertEqual(choice["finish_reason"], "tool_calls")
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "lookup")
        self.assertEqual(json.loads(call["function"]["arguments"]), {"query": "Paris", "limit": 2})

    def test_tool_result_and_tool_call_render_as_glm_multi_turn_conversation(self):
        req = {
            "reasoning_effort": "low",
            "tools": [{"type": "function", "function": {"name": "lookup", "description": "Look up a place",
                        "parameters": {"type": "object", "properties": {
                            "place": {"type": "string"}, "limit": {"type": "integer"}}}}}],
            "messages": [
                {"role": "user", "content": "Weather in Québec?"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "call-1", "type": "function",
                 "function": {"name": "lookup", "arguments": '{"place":"Québec","limit":2}'}}]},
                {"role": "tool", "tool_call_id": "call-1", "content": '{"temperature":12}'},
            ],
        }
        messages, tools, kwargs = openai_to_messages(req, family="glm")
        self.assertEqual(kwargs, {"reasoning_effort": "low"})
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call-1")
        self.assertEqual(messages[2]["tool_call_id"], "call-1")
        exported = Path(os.environ.get("STRATA_GLM_CHAT_TEMPLATE", ""))   # the GGUF's own template, when exported
        template_path = exported if exported.is_file() else ROOT / "serve/glm_chat_template.jinja"
        rendered = ChatTemplate(template_path).render(messages, tools=tools, **kwargs)
        self.assertTrue(rendered.startswith("[gMASK]<sop><|system|>Reasoning Effort: Low"), rendered[:100])
        self.assertIn("<|user|>Weather in Québec?", rendered)
        self.assertIn('<tool_call>lookup<arg_key>place</arg_key><arg_value>Québec</arg_value>'
                      '<arg_key>limit</arg_key><arg_value>2</arg_value></tool_call>', rendered)
        self.assertIn('<|observation|><tool_response>{"temperature":12}</tool_response>', rendered)
        self.assertTrue(rendered.endswith("<|assistant|><think>"), rendered[-80:])

    def test_glm_effort_mapping_and_stop_ids_do_not_change_qwen(self):
        self.assertEqual(openai_to_messages({"reasoning_effort": "medium"}, family="glm")[2],
                         {"reasoning_effort": "high"})
        # GLM cannot switch its thinking off: every spelling of "off" gets Low, the least it has; no effort = Max
        low = {"reasoning_effort": "low"}
        self.assertEqual(openai_to_messages({"reasoning_effort": "none"}, family="glm")[2], low)
        self.assertEqual(openai_to_messages({"reasoning": {"effort": "disabled"}}, family="glm")[2], low)
        self.assertEqual(openai_to_messages({"chat_template_kwargs": {"enable_thinking": False}}, family="glm")[2], low)
        self.assertEqual(openai_to_messages({"reasoning_effort": "none", "chat_template_kwargs":
                                             {"reasoning_effort": "high"}}, family="glm")[2], low)
        self.assertEqual(anthropic_to_messages({"thinking": {"type": "disabled"}}, family="glm")[2], low)
        self.assertEqual(openai_to_messages({}, family="glm")[2], {})
        self.assertEqual(openai_to_messages({"reasoning_effort": "max"}, family="glm")[2], {})
        self.assertEqual(anthropic_to_messages({"thinking": {"type": "disabled"}})[2], {"enable_thinking": False})
        self.assertEqual(openai_to_messages({"reasoning_effort": "none"})[2], {"enable_thinking": False})
        tok = ByteTokenizer()
        engine = MockEngine(tok, "answer")
        glm = Service(engine, tok, ChatTemplate(ROOT / "serve/glm_chat_template.jinja"), family="glm")
        qwen = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.assertEqual(glm.stop_ids, set(GLM_STOP_IDS))
        self.assertEqual(qwen.stop_ids, set(tok.encode("<|im_end|>", parse_special=True) +
                                             tok.encode("<|endoftext|>", parse_special=True)))
        prompt_ids, thinking, _ = glm.prepare([{"role": "user", "content": "hi"}], None,
                                             {"enable_thinking": False}, max_new=10)
        self.assertTrue(thinking)  # GLM's template always opens <think>, so the parser must start in reasoning.
        self.assertTrue(tok.decode(prompt_ids).endswith("<|assistant|><think>"))


if __name__ == "__main__":
    unittest.main()
