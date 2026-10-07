"""Offline task-detail presentation checks."""
import unittest

from langchain_core.messages import ToolMessage
from langgraph.types import Command
from rich.console import Console
from textual.app import App
from textual.widgets import Collapsible, Static

from app.tui import ToolRecord, ToolGroup


class ToolDetailsTests(unittest.IsolatedAsyncioTestCase):
    async def test_task_fields_markdown_and_raw_data_at_narrow_width(self):
        card = ToolRecord('task', {
            'subagent_type': 'researcher',
            'description': '只查官方资料\n返回来源\n[bold]保持字面文本[/bold]',
        })
        class Preview(App):
            def compose(self):
                yield card
        app = Preview()
        async with app.run_test(size=(48, 24)) as pilot:
            card.finish(Command(update={'messages': [ToolMessage(
                content='## Findings\n\n- 发现一\n- [Official](https://example.org)\n\n' + '资料' * 4100 + '\nEND',
                tool_call_id='task-1',
            )]}))
            card.collapsed = False
            await pilot.pause()
            rendered = '\n'.join(str(w.render()) for w in card.query(Static))
            self.assertIn('description\n只查官方资料\n返回来源', rendered)
            self.assertIn('[bold]保持字面文本[/bold]', rendered)
            console = Console(width=46, record=True)
            with console.capture() as capture:
                console.print(card.detail.render()._renderable)
            content = capture.get()
            self.assertIn('Findings', content)
            self.assertIn('END', content)
            self.assertNotIn('Command(update=', content)
            raw = card.query(Collapsible).first()
            self.assertTrue(raw.collapsed)
            raw.collapsed = False
            await pilot.pause()
            self.assertIn('Command(update=', str(card.raw_detail.render()))
            self.assertLessEqual(card.region.right, 48)

    async def test_group_collapses_counts_preserves_details_and_failure(self):
        group = ToolGroup()
        class Preview(App):
            def compose(self):
                yield group
        async with Preview().run_test(size=(48, 24)) as pilot:
            for name in ("ls", "read_file", "read_file"):
                card = ToolRecord(name, {"path": "/notes"})
                await group.add_record(card)
                card.finish("data")
            group.refresh_summary()
            self.assertTrue(group.collapsed)
            self.assertIn("Read 2", group.title)
            self.assertIn("List 1", group.title)
            self.assertIn("3 completed", group.title)
            group.collapsed = False
            await pilot.pause()
            self.assertEqual(len(group.query(ToolRecord)), 3)
            group.records[-1].finish("failure", state="失败")
            group.refresh_summary()
            self.assertIn("1 failed", group.title)
            self.assertNotIn("3 completed", group.title)

    async def test_task_error_and_plain_result(self):
        card = ToolRecord('task', {})
        card.finish(Command(update={'messages': [ToolMessage(
            content='Search unavailable', tool_call_id='t', status='error')]}))
        self.assertIn('失败', card.title)
        card.finish('Unknown subagent', state='失败')
        self.assertIn('失败', card.title)


if __name__ == '__main__':
    unittest.main()
